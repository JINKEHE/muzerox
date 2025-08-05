# Python libraries
import functools
import collections
import time
from typing import Callable
import pickle

# Hyperparameters
from omegaconf import OmegaConf

# Jax libraries
import jax
import jax.numpy as jnp
import chex
import mctx
import optax
import rlax
import flax
from flax.core.frozen_dict import FrozenDict
import flashbax as fbx
from flashbax.buffers import sum_tree

# Utils
from .utils import transform_to_2hot, compute_kl_divergence_with_probs_and_logits, compute_entropy_with_probs

# Neural network architecture
from . import nets

TimeStep = collections.namedtuple('TimeStep', [
    'o_t',
    'a_t',
    'r_t',
    'd_t',
    't_t',
    'v_t', # MCTS value
    'pi_t' # Policy value
])

class MuZeroAgent:
    """A MuZero agent implementation.
    
    """
    class MuZeroModel:
        """Internal model class for MuZero's world model and predictions.
        
        """
        def __init__(self, net: flax.linen.Module, discount_factor: float, logits_to_scalar: Callable):
            self.net = net
            self.discount_factor = discount_factor
            self.logits_to_scalar = logits_to_scalar

        def init_fn(self, params: dict, stacked_obs: chex.Array):
            """Initial state encoding and predictions."""
            encoded_state = self.net.apply(params, stacked_obs, method=self.net.encode)
            _, value_logits, policy_prior_logits = self.net.apply(
                params, encoded_state, method=self.net.predict
            )
            value = self.logits_to_scalar(value_logits)
            return mctx.RootFnOutput(
                prior_logits=policy_prior_logits,
                value=value,
                embedding=encoded_state
            )

        def recurrent_fn(self, params: dict, rng_key: chex.PRNGKey, action: chex.Array, encoded_state: chex.Array):
            """State transition and predictions."""
            next_encoded_state = self.net.apply(
                params, encoded_state, action, method=self.net.step
            )
            reward_logits, value_logits, policy_prior_logits = self.net.apply(
                params, next_encoded_state, method=self.net.predict
            )
            reward = self.logits_to_scalar(reward_logits)
            value = self.logits_to_scalar(value_logits)
            discount = jnp.ones_like(reward) * self.discount_factor
            return mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=policy_prior_logits,
                value=value
            ), next_encoded_state

    @flax.struct.dataclass
    class AgentState:
        """Container for agent state."""
        env_step: int
        learn_step: int
        gradient_step: int
        reanalyze_step: int
        params: dict
        opt_state: optax.TraceState
        replay_buffer_state: fbx.prioritised_trajectory_buffer.BufferState

    def __init__(self, config: OmegaConf):
        """Initialize the MuZero agent with the given configuration."""
        
        # Initialize configuration parameters
        self.config = config
        self.agent_config = config['agent']
        self.act_config = self.agent_config['act']
        self.learn_config = self.agent_config['learn']
        self.optimizer_config = config['agent']['learn']['optimizer']
        self.replay_buffer_config = config['agent']['learn']['replay_buffer']
        self.net_config = self.agent_config['neural_network']
        self.loss_config = self.learn_config['loss']
        self.env_config = self.config['env']
        self.reanalyze_config = self.learn_config['reanalyze']

        # Initialize the replay buffer
        self.num_envs = self.env_config['train']['num_envs']
        self.n_stack = self.env_config['common']['n_stack']
        self.unroll_steps = self.loss_config['model']['unroll_steps']
        self.td_steps = self.loss_config['value']['td_steps']
        self.num_actions = self.env_config['common']['num_actions']
        self.sequence_length = (
            self.n_stack - 1 +  # observations for stacking
            self.unroll_steps +  # steps to learn model
            self.td_steps + 1  # steps for value targets
        )
        self.n_skip = self.env_config['common']['n_skip']
        self.batch_size = self.optimizer_config['batch_size']
        self.replay_buffer_max_size = self.replay_buffer_config['max_size']
        self.replay_buffer_min_size = self.replay_buffer_config['min_size']
        assert self.replay_buffer_max_size % self.num_envs == 0
        assert self.replay_buffer_min_size % self.num_envs == 0
        min_length_time_axis = (self.replay_buffer_min_size//self.num_envs)+self.n_stack-1
        max_length_time_axis = (self.replay_buffer_max_size//self.num_envs)+self.n_stack-1
        self.replay_buffer = fbx.make_prioritised_trajectory_buffer(
            add_batch_size=self.num_envs,
            sample_batch_size=self.batch_size,
            sample_sequence_length=self.sequence_length,
            period=1,
            min_length_time_axis=min_length_time_axis,
            max_length_time_axis=max_length_time_axis,
            priority_exponent=self.replay_buffer_config['PER']['alpha'],
            device="gpu"
        )
        # Do we want to save the replay buffer as part of the agent state?
        self.save_replay_buffer = self.replay_buffer_config['save']

        # Initialize the neural network architecture
        self.net = nets.EfficientZeroNet(
            num_actions=self.config['env']['common']['num_actions'],
            num_bins=self.agent_config['value_transformation']['num_bins'],
            **self.net_config
        )

        # Initialize the lr schedule
        online_learn_steps = ((self.replay_buffer_max_size - self.replay_buffer_min_size) // self.num_envs) // self.config['training']['learn_per_update_step']
        offline_learn_steps = self.config['training']['offline_update_steps']
        self.learn_steps = online_learn_steps + offline_learn_steps
        self.gradient_steps = int(self.learn_steps * self.optimizer_config['steps'])
        self.warmup_gradient_steps = int(self.gradient_steps * self.optimizer_config['warmup_ratio'])
        print("total gradient steps", self.gradient_steps)
        print("warmup gradient steps", self.warmup_gradient_steps)
        if self.warmup_gradient_steps == 0:
            self.lr_schedule = optax.constant_schedule(
                self.optimizer_config["lr"]
            )
        else:
            self.lr_schedule = optax.linear_schedule(
                init_value=0.0,
                end_value=self.optimizer_config["lr"],
                transition_steps=self.warmup_gradient_steps,
                transition_begin=0
            )
        print("learn_steps", self.learn_steps)
        
        # Initialize the optimizer
        # Avoid weight decay on bias and layer norm parameters
        def make_decay_mask(params):
            mask = {}
            for key, subtree in params.items():
                if isinstance(subtree, (dict, FrozenDict)):
                    mask[key] = make_decay_mask(subtree)
                else:
                    mask[key] = (key == "kernel")
            return mask
        self.tx = optax.chain(
            optax.clip_by_global_norm(self.optimizer_config['max_grad_norm']),
            optax.adamw(
                learning_rate=self.lr_schedule, 
                weight_decay=self.optimizer_config["weight_decay"],
                mask=make_decay_mask
            )
        )
        self.sgd_steps = self.optimizer_config['steps']

        # Initialize value transformation functions
        self.num_bins = self.agent_config['value_transformation']['num_bins']
        self.support = self.agent_config['value_transformation']['support']
        self.min_value = self.support[0]
        self.max_value = self.support[1]
        self.use_transformation = self.agent_config['value_transformation']['use_transformation']
        if self.use_transformation:
            value_transformation = rlax.signed_hyperbolic
            inverse_value_transformation = rlax.signed_parabolic
        else:
            value_transformation = lambda x: x
            inverse_value_transformation = lambda x: x
        self.scalar_to_probs = lambda x: transform_to_2hot(
            value_transformation(x), 
            min_value=self.min_value, 
            max_value=self.max_value, 
            num_bins=self.num_bins
        )
        self.probs_to_scalar = lambda x: inverse_value_transformation(
            rlax.transform_from_2hot(x, min_value=self.min_value, max_value=self.max_value, num_bins=self.num_bins)
        )
        self.logits_to_scalar = lambda x: inverse_value_transformation(
            rlax.transform_from_2hot(jax.nn.softmax(x, axis=-1), min_value=self.min_value, max_value=self.max_value, num_bins=self.num_bins)
        )

        # Initialize the qtransform function used in MCTS
        by = self.act_config['mcts']['qtransform']['by']
        epsilon = self.act_config['mcts']['qtransform']['epsilon']
        if by == "updatable_min_max":
            qtransform_fn = functools.partial(mctx.qtransform_by_updatable_min_max, epsilon=epsilon)
        elif by == "parent_and_siblings":
            qtransform_fn = functools.partial(mctx.qtransform_by_parent_and_siblings, epsilon=epsilon)
        else:
            raise ValueError(f"Unknown qtransform: {by}")
        self.act_config_train = dict(self.act_config['train']) | dict(self.act_config['mcts']) | {"qtransform": qtransform_fn}
        self.act_config_eval = dict(self.act_config['eval']) | dict(self.act_config['mcts']) | {"qtransform": qtransform_fn}
        self.act_config_reanalyze = dict(self.act_config['mcts']) | {"qtransform": qtransform_fn}
    
        # Initialize the MuZero model
        self.discount_factor = self.agent_config['discount_factor'] ** self.n_skip
        self.model = self.MuZeroModel(
            net=self.net,
            discount_factor=self.discount_factor,
            logits_to_scalar=self.logits_to_scalar
        )
        
        # Initialize reanalyze parameters
        if 'batch_size' not in self.reanalyze_config or self.reanalyze_config['batch_size'] is None:
            vram = jax.devices()[0].memory_stats()['bytes_limit'] // (1024**3)
            if vram >= 30:
                self.reanalyze_batch_size = 5000
            else:
                self.reanalyze_batch_size = 2000
        else:
            self.reanalyze_batch_size = self.reanalyze_config['batch_size']
        print(f"reanalyze_batch_size: {self.reanalyze_batch_size}")
        self.reanalyze_batch_size_time_axis = self.reanalyze_batch_size // self.num_envs if self.reanalyze_batch_size is not None else None
        self.reanalyze_per_learn_step = self.reanalyze_config['per_learn_step']
        
        # Profile or not
        self.profile = self.config['logging']['profiling']
        
    def save(self, agent_state: AgentState, checkpoint_dir: str, checkpoint_name: str):
        """Save agent state to checkpoint."""
        if self.save_replay_buffer == False:
            agent_state = agent_state.replace(replay_buffer_state=None)
        with open(f"{checkpoint_dir}/{checkpoint_name}", "wb") as f:
            pickle.dump(agent_state, f)

    def restore(self, checkpoint_dir: str, checkpoint_name: str) -> AgentState:
        """Restore agent state from checkpoint."""
        with open(f"{checkpoint_dir}/{checkpoint_name}", "rb") as f:
            agent_state = pickle.load(f)
        return agent_state

    def init_state(self, rng_key: chex.PRNGKey, state: chex.Array) -> AgentState:
        """Initialize agent state."""
        
        # Initialize neural network parameters
        rng_key, net_init_key = jax.random.split(rng_key)
        params = self.net.init(net_init_key, state, jnp.ones((state.shape[0], )))
        
        # Initialize optimizer state
        opt_state = self.tx.init(params)
        
        # Initialize replay buffer state
        initial_timestep = TimeStep(
            o_t=state[:,:,:,-3:].astype(jnp.uint8),
            a_t=jnp.zeros((self.num_envs,), dtype=jnp.int32),
            r_t=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            d_t=jnp.zeros((self.num_envs,), dtype=jnp.bool),
            t_t=jnp.zeros((self.num_envs,), dtype=jnp.bool),
            v_t=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            pi_t=jnp.zeros((self.num_envs, self.num_actions), dtype=jnp.float32)
        )
        replay_buffer_state = self.replay_buffer.init(jax.tree.map(lambda x: x[0], initial_timestep))

        # Add self.n_stack-1 frames to the replay buffer
        for _ in range(self.n_stack-1):
            replay_buffer_state = self.replay_buffer.add(
                replay_buffer_state, 
                jax.tree.map(lambda x: x[:,None], initial_timestep)
            )

        return self.AgentState(
            env_step=0,
            learn_step=0,
            gradient_step=0,
            reanalyze_step=0,
            params=params,
            opt_state=opt_state,
            replay_buffer_state=replay_buffer_state
        )

    def _temperature_fn(self, env_step: int, schedule: list[tuple[float, float]]) -> float:
        """Compute temperature for action selection during training."""
        progress = env_step / self.config['training']['env_steps']
        for threshold, temperature in schedule:
            if progress < threshold:
                return temperature
        return float(schedule[-1][1])
    
    def _beta_fn(self, learn_step: int) -> float:
        """Compute beta for prioritized experience replay."""
        start = self.learn_config['replay_buffer']['PER']['beta_start']
        end = self.learn_config['replay_buffer']['PER']['beta_end']
        progress = learn_step / self.learn_steps
        return start + progress * (end - start)
        
    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _observe(
        self,
        replay_buffer_state: fbx.trajectory_buffer.BufferState, 
        stacked_obs: chex.Array, 
        action: chex.Array, 
        value: chex.Array,
        policy: chex.Array,
        reward: chex.Array, 
        terminated: chex.Array, 
        truncated: chex.Array,
    ) -> fbx.trajectory_buffer.BufferState:
        """Add experience to replay buffer."""
        timestep = TimeStep(
            o_t=stacked_obs[:,:,:,-3:].astype(jnp.uint8), # Only add the latest observation to save memory
            a_t=action,
            r_t=reward,
            d_t=terminated,
            t_t=truncated,
            v_t=value,
            pi_t=policy,
        )
        return self.replay_buffer.add(
            replay_buffer_state, 
            jax.tree.map(lambda x: x[:,None], timestep)
        )

    def observe(
        self, 
        agent_state: AgentState, 
        stacked_obs: chex.Array,
        action: chex.Array, 
        policy_output: dict,
        reward: chex.Array, 
        terminated: chex.Array, 
        truncated: chex.Array,
        logs: dict = {}
    ) -> AgentState:
        """Add experience to replay buffer and update agent state."""
        return self.AgentState(
            env_step=agent_state.env_step + stacked_obs.shape[0],
            learn_step=agent_state.learn_step,
            gradient_step=agent_state.gradient_step,
            reanalyze_step=agent_state.reanalyze_step,
            params=agent_state.params,
            opt_state=agent_state.opt_state,
            replay_buffer_state=self._observe(
                agent_state.replay_buffer_state,
                stacked_obs.astype(jnp.uint8), 
                action, 
                policy_output['value'], 
                policy_output['policy'], 
                reward, 
                terminated, 
                truncated
            )
        )
    
    @functools.partial(jax.jit, static_argnums=(0,4,5,6,7,8,9,10,11))
    def _act(
        self, 
        rng_key: chex.PRNGKey, 
        params: dict, 
        stacked_obs: chex.Array, 
        num_simulations: int,
        pb_c_init: float,
        pb_c_base: int,
        dirichlet_fraction: float, 
        dirichlet_alpha: float, 
        max_depth: int,
        temperature: float,
        qtransform: Callable
    ):
        """Run MCTS to select an action."""
        root = self.model.init_fn(params, stacked_obs.astype(jnp.float32))
        policy_output: mctx.PolicyOutput = mctx.muzero_policy(
            params=params,
            rng_key=rng_key,
            root=root,
            recurrent_fn=self.model.recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=None,
            max_depth=max_depth,
            loop_fn=jax.lax.fori_loop,
            dirichlet_fraction=dirichlet_fraction,
            dirichlet_alpha=dirichlet_alpha,
            qtransform=qtransform,
            pb_c_init=pb_c_init,
            pb_c_base=pb_c_base,
            temperature=temperature
        )
        search_tree = policy_output.search_tree
        # compute average search tree depth
        parent_indices = jnp.copy(search_tree.parents)
        # shape: (num_envs, num_simulations+1)
        depths = jnp.zeros_like(parent_indices)
        
        # Use JAX's functional approach instead of in-place modification
        # Initialize with zeros and then update using a scan function
        def update_depths(depths, i):
            # Fix the broadcasting issue by ensuring dimensions match
            # Get the parent indices for the current simulation step
            parent_idx = parent_indices[:, i]
            # Get the depths of the parents
            parent_depths = jnp.take_along_axis(depths, parent_idx[:, None], axis=1)
            # Add 1 to the parent depths and update the current depths
            new_depths = depths.at[:, i].set(1 + parent_depths[:, 0])
            return new_depths, None
        
        # Apply the update for each simulation step
        depths, _ = jax.lax.scan(update_depths, depths, jnp.arange(1, num_simulations+1))
        
        # compute average depth
        avg_depth = jnp.mean(depths)
        avg_max_value = jnp.mean(policy_output.search_tree.extra_data.max_value)
        avg_min_value = jnp.mean(policy_output.search_tree.extra_data.min_value)

        value_and_policy = {
            "value": policy_output.search_tree.summary().value, # MCTS value
            "policy": policy_output.action_weights, # MCTS policy
            "predicted_value": root.value # Predicted value            
        }
        entropy = compute_entropy_with_probs(policy_output.action_weights)
        logs = {
            "avg_depth": avg_depth,
            "avg_max_value": avg_max_value,
            "avg_min_value": avg_min_value,
            "avg_normalized_visit_entropy": entropy.mean() / jnp.log(self.num_actions)
        }
        return policy_output.action, value_and_policy, logs

    def act(
        self, 
        rng_key: chex.PRNGKey, 
        agent_state: AgentState, 
        stacked_obs: chex.Array, 
        logs: dict = {"Training": {}}, 
        **kwargs
    ):
        """Select an action using MCTS planning.
        
        Args:
            rng_key: Random key for stochastic operations
            agent_state: Current agent state
            stacked_obs: Stacked observations tensor
            **kwargs: Additional arguments passed to MCTS
        
        Returns:
            action: Selected action
            policy_output: Full policy information from MCTS
        """
        temperature = self._temperature_fn(agent_state.env_step, self.act_config_train['temperature'])
        mcts_kwargs = self.act_config_train | {"temperature": temperature} | kwargs
        mcts_key, randomize_key = jax.random.split(rng_key, 2)
        action, policy_output, act_logs = self._act(
            mcts_key, 
            agent_state.params, 
            stacked_obs, 
            **mcts_kwargs
        )
        action = self._randomize_action_if_replay_buffer_is_small(randomize_key, agent_state.replay_buffer_state, action)
        logs['Training']['temperature'] = temperature
        logs['Training'].update(act_logs)
        logs['Training']['mean_mcts_value'] = policy_output['value'].mean()
        logs['Training']['max_mcts_value'] = policy_output['value'].max()
        logs['Training']['min_mcts_value'] = policy_output['value'].min()

        return action, policy_output
    
    def eval_act(
        self,
        rng_key: chex.PRNGKey,
        agent_state: AgentState,
        stacked_obs: chex.Array,
        logs: dict = {'Evaluation': {}},
        **kwargs
    ):
        """Select action for evaluation (no exploration)."""
        temperature = self._temperature_fn(agent_state.env_step, self.act_config_eval['temperature'])
        mcts_kwargs = self.act_config_eval | {"temperature": temperature} | kwargs
        action, policy_output, _ = self._act(
            rng_key, 
            agent_state.params, 
            stacked_obs, 
            **mcts_kwargs
        )
        logs['Evaluation']['temperature'] = temperature
        return action, policy_output
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def eval_act_without_mcts(self, rng_key: chex.PRNGKey, agent_state: AgentState, stacked_obs: chex.Array):
        """Select action for evaluation (no MCTS)."""
        # encode the stacked observations into a latent state
        latent_state = self.net.apply(
            agent_state.params,
            stacked_obs,
            method=self.net.encode
        )
        # predict the value and policy
        _, value_logits, policy_logits = self.net.apply(
            agent_state.params,
            latent_state,
            method=self.net.predict
        )
        # select the action with the highest policy probability
        action = jax.random.categorical(rng_key, policy_logits)
        return action, {"value": self.logits_to_scalar(value_logits), "policy": policy_logits}
    
    def _stack_observations(self, observations_slice, resets_slice):
        """Stack observations with reset handling."""
        new_stacked_obs = []
        new_stacked_obs.append(observations_slice[-1])
        use_prev = jnp.ones_like(resets_slice[-1])
        for j in range(1, self.n_stack):
            use_prev = use_prev * (1-resets_slice[-j])
            prev_obs = jnp.where(
                use_prev[:, None, None, None],
                observations_slice[-(j+1)],
                new_stacked_obs[-1]
            )
            new_stacked_obs.append(prev_obs)
        new_stacked_obs = jnp.stack(new_stacked_obs[::-1], axis=0)
        new_stacked_obs = jnp.moveaxis(new_stacked_obs, 0, -2)
        new_stacked_obs = new_stacked_obs.reshape(-1, 96, 96, 12)
        return new_stacked_obs
    
    def _compute_model_predictions(self, params, root_stacked_observations, actions):
        """Compute model predictions for the given sequence."""
        # Encode, unroll and predict
        reward_logits, value_logits, policy_logits = self.net.apply(
            params, 
            root_stacked_observations, 
            actions, 
            method=self.net.encode_and_unroll_and_predict
        )
        chex.assert_axis_dimension(reward_logits, axis=0, expected=self.unroll_steps+1)
        chex.assert_axis_dimension(value_logits, axis=0, expected=self.unroll_steps+1)
        chex.assert_axis_dimension(policy_logits, axis=0, expected=self.unroll_steps+1)
        return reward_logits, value_logits, policy_logits

    def _compute_reward_loss(self, reward_logits, target_rewards, terminations, truncations):
        """Compute reward prediction loss."""

        # Assert shapes
        batch_size = reward_logits.shape[1]
        chex.assert_shape(reward_logits, (self.unroll_steps, batch_size, self.num_bins))
        chex.assert_shape(target_rewards, (self.unroll_steps, batch_size))
        chex.assert_shape(terminations, (self.unroll_steps-1, batch_size))
        chex.assert_shape(truncations, (self.unroll_steps-1, batch_size))

        # Make all targets 0 after terminations
        not_terminated = jnp.concatenate([
            jnp.ones((1, terminations.shape[1])),
            jnp.cumprod(1-terminations, axis=0)
        ])
        target_rewards = target_rewards * not_terminated
        target_reward_probs = self.scalar_to_probs(target_rewards)
        chex.assert_shape(target_reward_probs, (self.unroll_steps, batch_size, self.num_bins))
        
        # Compute loss
        loss = compute_kl_divergence_with_probs_and_logits(
            target_reward_probs, 
            reward_logits
        )
        chex.assert_shape(loss, (self.unroll_steps, batch_size))

        # Mask out losses after truncations
        not_truncated = jnp.concatenate([
            jnp.ones((1, truncations.shape[1])),
            jnp.cumprod(1-truncations, axis=0)
        ])
        masks = not_truncated / not_truncated.sum(axis=0, keepdims=True)
        masked_reward_loss = (loss * masks).sum(axis=0)
        chex.assert_shape(masked_reward_loss, (batch_size,))
        unmasked_reward_loss = loss.mean(axis=0)
        chex.assert_shape(unmasked_reward_loss, (batch_size,))

        logs = {
            'mean_target_rewards': jnp.mean(target_rewards),
            'mean_predicted_rewards': jnp.mean(self.logits_to_scalar(reward_logits)),
            'min_target_rewards': jnp.min(target_rewards),
            'max_target_rewards': jnp.max(target_rewards),
            'min_predicted_rewards': jnp.min(self.logits_to_scalar(reward_logits)),
            'max_predicted_rewards': jnp.max(self.logits_to_scalar(reward_logits)),
            'reward_loss': jnp.mean(masked_reward_loss),
        }
        return masked_reward_loss, logs

    def _compute_target_values(self, rewards, terminations, truncations, values):
        batch_size = rewards.shape[1]
        chex.assert_shape(rewards, (self.unroll_steps+self.td_steps, batch_size))
        chex.assert_shape(terminations, (self.unroll_steps+self.td_steps, batch_size))
        chex.assert_shape(truncations, (self.unroll_steps+self.td_steps, batch_size))
        chex.assert_shape(values, (self.unroll_steps+self.td_steps+1, batch_size))
        
        # Compute target values
        target_values = []
        for i in range(self.unroll_steps+1):
            if self.td_steps == 0:
                target_value = values[i]
            else:
                discount_factor = jnp.ones((values.shape[1], ), dtype=jnp.float32)
                continue_factor = jnp.ones((values.shape[1], ), dtype=jnp.float32)
                target_value = jnp.zeros((values.shape[1], ), dtype=jnp.float32)
                for j in range(self.td_steps):
                    target_value += discount_factor * continue_factor * rewards[i+j]
                    # bootstrap if 
                    # case 1. still continuing but now terminated
                    # case 2. still continuing but now reaching the end
                    # case 3. still continuing but truncated
                    # in case 1 and 3, we bootstrap
                    # otherwise, we don't bootstrap
                    bootstrap_here = continue_factor * jnp.maximum(jnp.maximum(jnp.ones_like(terminations[i+j]) * (j==self.td_steps-1), terminations[i+j]), truncations[i+j])
                    bootstrap_value = jnp.where(terminations[i+j], jnp.zeros_like(values[i+j+1]), values[i+j+1])
                    discount_factor *= self.discount_factor 
                    continue_factor *= (1-terminations[i+j]) * (1-truncations[i+j])
                    target_value += discount_factor * bootstrap_here * bootstrap_value
            target_values.append(target_value)
        target_values = jnp.stack(target_values, axis=0)

        # Make all targets 0 after terminations
        continuing = jnp.concatenate([
            jnp.ones((1, terminations.shape[1])),
            jnp.cumprod(1-terminations[:self.unroll_steps], axis=0)
        ])
        target_values = target_values * continuing
        
        return target_values
    
    def _compute_value_loss(self, value_logits, rewards, terminations, truncations, values):
        """Compute value prediction loss."""

        # Assert shapes
        batch_size = value_logits.shape[1]
        chex.assert_shape(value_logits, (self.unroll_steps+1, batch_size, self.num_bins))
        chex.assert_shape(values, (self.unroll_steps+self.td_steps+1, batch_size))
        chex.assert_shape(rewards, (self.unroll_steps+self.td_steps, batch_size))
        chex.assert_shape(terminations, (self.unroll_steps+self.td_steps, batch_size))
        chex.assert_shape(truncations, (self.unroll_steps+self.td_steps, batch_size))

        # Compute target values
        target_values = self._compute_target_values(rewards, terminations, truncations, values)
        target_value_probs = self.scalar_to_probs(target_values)
        chex.assert_shape(target_value_probs, (self.unroll_steps+1, batch_size, self.num_bins))

        # Compute loss
        value_loss = compute_kl_divergence_with_probs_and_logits(
            jax.lax.stop_gradient(target_value_probs),
            value_logits
        )
        chex.assert_shape(value_loss, (self.unroll_steps+1, batch_size))

        # Mask out losses after truncations
        not_truncated = jnp.concatenate([
            jnp.ones((1, truncations.shape[1])),
            jnp.cumprod(1-truncations[:self.unroll_steps], axis=0)
        ])
        chex.assert_shape(not_truncated, (self.unroll_steps+1, batch_size))
        truncation_masks = not_truncated / not_truncated.sum(axis=0, keepdims=True)
        masked_value_loss = (value_loss * truncation_masks).sum(axis=0)
        chex.assert_shape(masked_value_loss, (batch_size,))
        unmasked_value_loss = value_loss.mean(axis=0)
        chex.assert_shape(unmasked_value_loss, (batch_size,))

        predicted_values = self.logits_to_scalar(value_logits)

        logs = {
            'mean_target_values': jnp.mean(target_values),
            'max_target_values': jnp.max(target_values),
            'min_target_values': jnp.min(target_values),
            'mean_predicted_values': jnp.mean(predicted_values),
            'max_predicted_values': jnp.max(predicted_values),
            'min_predicted_values': jnp.min(predicted_values),
            'value_loss': jnp.mean(masked_value_loss)
        }
        return value_loss, target_values, logs

    def _compute_policy_loss(self, policy_logits, target_policy_probs, actions, terminations, truncations):
        """Compute policy prediction loss."""

        # Assert shapes
        batch_size = policy_logits.shape[1]
        chex.assert_shape(policy_logits, (self.unroll_steps+1, batch_size, self.num_actions))
        chex.assert_shape(target_policy_probs, (self.unroll_steps+1, batch_size, self.num_actions))
        chex.assert_shape(actions, (self.unroll_steps+1, batch_size))

        not_terminated = jnp.concatenate([
            jnp.ones((1, terminations.shape[1])),
            jnp.cumprod(1-terminations, axis=0)
        ])[:, :, None]
        chex.assert_shape(not_terminated, (self.unroll_steps+1, batch_size, 1))
        
        # Set target probs to actions taken after terminations
        one_hot_actions = jax.nn.one_hot(actions, self.num_actions)
        target_policy_probs = target_policy_probs * not_terminated + (1-not_terminated) * one_hot_actions
        chex.assert_shape(target_policy_probs, (self.unroll_steps+1, batch_size, self.num_actions))

        # Compute KL divergence
        policy_loss = compute_kl_divergence_with_probs_and_logits(
            jax.lax.stop_gradient(target_policy_probs),
            policy_logits
        )

        # Compute entropy
        predicted_policy = jax.nn.softmax(policy_logits)
        predicted_policy_entropy = compute_entropy_with_probs(predicted_policy)
        target_policy_entropy = compute_entropy_with_probs(target_policy_probs)
        entropy_loss = -predicted_policy_entropy

        # Integrate losses
        policy_loss = policy_loss + self.learn_config['loss']['policy']['entropy_coef'] * entropy_loss
        
        # Mask out losses after truncations
        not_truncated = jnp.concatenate([
            jnp.ones((1, truncations.shape[1])),
            jnp.cumprod(1-truncations, axis=0)
        ])
        chex.assert_shape(not_truncated, (self.unroll_steps+1, batch_size))
        masks = not_truncated / not_truncated.sum(axis=0, keepdims=True)
        chex.assert_shape(policy_loss, (self.unroll_steps+1, batch_size))
        masked_policy_loss = (policy_loss * masks).sum(axis=0)
        chex.assert_shape(masked_policy_loss, (batch_size,))
        unmasked_policy_loss = policy_loss.mean(axis=0)
        chex.assert_shape(unmasked_policy_loss, (batch_size,))

        logs = {
            "policy_loss": jnp.mean(masked_policy_loss),
            "unmasked_policy_loss": jnp.mean(policy_loss),
            "target_policy_entropy": jnp.mean(target_policy_entropy),
            "predicted_policy_entropy": jnp.mean(predicted_policy_entropy),
            "entropy_loss": jnp.mean(entropy_loss)
        }
        return masked_policy_loss, logs
    
    def _compute_losses(self, params: dict, experience, importance_weights):
        """Compute MuZero losses for a batch of experiences.
        
        Returns:
            total_loss: Combined loss value
            (new_priorities, logs): Metrics and debug information
        """
        logs = {}

        # Prepare batch data
        observations = experience.o_t
        terminations = experience.d_t
        truncations = experience.t_t
        rewards = experience.r_t
        actions = experience.a_t
        values = experience.v_t
        policies = experience.pi_t

        # Get stacked observations - essentially the root state
        resets = jnp.logical_or(terminations, truncations)
        stacked_obs = self._stack_observations(observations[:self.n_stack], resets[:self.n_stack-1])

        # Unroll the model and compute model predictions
        reward_logits, value_logits, policy_logits = self._compute_model_predictions(
            params, 
            stacked_obs, 
            actions[self.n_stack-1:self.n_stack-1+self.unroll_steps]
        )

        # Reward loss
        reward_loss, reward_logs = self._compute_reward_loss(
            reward_logits[1:], 
            rewards[self.n_stack-1:self.n_stack-1+self.unroll_steps], 
            terminations[self.n_stack-1:self.n_stack-1+self.unroll_steps-1],
            truncations[self.n_stack-1:self.n_stack-1+self.unroll_steps-1]
        )
        logs.update(reward_logs)
        
        # Value loss
        chex.assert_axis_dimension(value_logits, axis=0, expected=self.unroll_steps+1)
        value_loss, target_values, value_logs = self._compute_value_loss(
            value_logits, # unroll_steps+1
            rewards[self.n_stack-1:self.n_stack-1+self.unroll_steps+self.td_steps], 
            terminations[self.n_stack-1:self.n_stack-1+self.unroll_steps+self.td_steps], 
            truncations[self.n_stack-1:self.n_stack-1+self.unroll_steps+self.td_steps],
            values[self.n_stack-1:self.n_stack-1+self.unroll_steps+self.td_steps+1] 
        )
        logs.update(value_logs)

        # Policy loss
        target_policy_probs = policies[self.n_stack-1:self.n_stack-1+self.unroll_steps+1]
        chex.assert_shape(target_policy_probs, (self.unroll_steps+1, self.batch_size, self.num_actions))
        policy_loss, policy_logs = self._compute_policy_loss(
            policy_logits, 
            target_policy_probs, 
            actions[self.n_stack-1:self.n_stack-1+self.unroll_steps+1],
            terminations[self.n_stack-1:self.n_stack-1+self.unroll_steps],
            truncations[self.n_stack-1:self.n_stack-1+self.unroll_steps]
        )
        logs.update(policy_logs)

        # Combine losses
        total_loss = reward_loss * self.learn_config['loss']['reward']['coef'] + \
                     policy_loss * self.learn_config['loss']['policy']['coef'] + \
                     value_loss * self.learn_config['loss']['value']['coef']
        chex.assert_shape(importance_weights, (self.batch_size,))
        weighted_loss = (total_loss * importance_weights).mean()
        logs['unweighted_loss'] = total_loss.mean()
        logs['weighted_loss'] = weighted_loss

        # Compute new priorities
        priority = self.learn_config['replay_buffer']['PER']['priority']
        predicted_values = self.logits_to_scalar(value_logits)
        predicted_values_at_root = predicted_values[0]
        target_values_at_root = target_values[0]
        mcts_values_at_root = values[self.n_stack-1]
        if priority == 'predicted_to_target':
            new_priorities = jnp.abs(target_values_at_root - predicted_values_at_root)
        elif priority == 'predicted_to_mcts':
            new_priorities = jnp.abs(mcts_values_at_root - predicted_values_at_root)
        elif priority == 'mcts_to_target':
            new_priorities = jnp.abs(target_values_at_root - mcts_values_at_root)
        else:
            raise ValueError(f"Invalid priority: {priority}")
        chex.assert_shape(new_priorities, (self.batch_size,))

        return weighted_loss, (new_priorities, logs)
    
    @functools.partial(jax.jit, static_argnums=(0), donate_argnums=(2,3,4))
    def _learn(self, rng_key, params, opt_state, replay_buffer_state, beta):
        """Internal learning function implementing the MuZero learning algorithm.
        
        This function:
        1. Samples trajectories from replay buffer
        2. Computes value/policy/reward predictions
        3. Calculates losses 
        4. Updates network parameters
        """

        def train_step(carry, _):
            """Single training step."""
            rng_key, params, opt_state, replay_buffer_state = carry

            # Sample batch and compute losses
            rng_key, subkey = jax.random.split(rng_key)
            batch = self.replay_buffer.sample(replay_buffer_state, subkey)
            experience = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).astype(jnp.float32), batch.experience)
            
            # Log initial priorities
            logs = {}
            importance_weights = (1. / batch.probabilities).astype(jnp.float32)
            importance_weights **= beta
            importance_weights = jnp.where(batch.probabilities == 0.0, 0.0, importance_weights)
            importance_weights /= jnp.max(importance_weights)
            logs['importance_ratio_max'] = jnp.max(importance_weights)
            logs['importance_ratio_min'] = jnp.min(importance_weights)
            logs['importance_ratio_mean'] = jnp.mean(importance_weights)
            
            # Update parameters
            grads, (new_priorities, loss_logs) = jax.grad(self._compute_losses, argnums=0, has_aux=True)(params, experience, importance_weights)
            updates, new_opt_state = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            # Log priorities before update
            logs['new_priorities_mean'] = jnp.mean(new_priorities)
            logs['new_priorities_max'] = jnp.max(new_priorities)
            logs['new_priorities_min'] = jnp.min(new_priorities)

            # Update priorities
            new_priorities = jnp.clip(new_priorities, min=1e-6)
            new_priorities = jnp.where(batch.probabilities == 0.0, 0.0, new_priorities)
            new_replay_buffer_state = self.replay_buffer.set_priorities(
                replay_buffer_state,
                batch.indices,
                jax.lax.stop_gradient(new_priorities)
            )
            
            # Generate additional logs
            logs['grad_norm'] = jnp.linalg.norm(
                jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(grads)])
            )
            
            logs.update(loss_logs)
            return (rng_key, new_params, new_opt_state, new_replay_buffer_state), logs

        (_, new_params, new_opt_state, new_replay_buffer_state), logs = jax.lax.scan(
            f=train_step,
            init=(rng_key, params, opt_state, replay_buffer_state),
            xs=jnp.arange(self.sgd_steps)
        )

        # Average metrics across steps
        logs = jax.tree.map(lambda x: jnp.mean(jnp.array(x), axis=0), logs)

        return new_params, new_opt_state, new_replay_buffer_state, logs
    
    @functools.partial(jax.jit, static_argnums=(0), donate_argnums=(3,))
    def _reanalyze_batch(self, rng_key, params, replay_buffer_state, minibatch_start_index):
        """Reanalyze a batch of experiences."""

        # Prepare data
        batch = jax.tree.map(lambda x: jax.lax.dynamic_slice_in_dim(x, minibatch_start_index*self.reanalyze_batch_size_time_axis, self.reanalyze_batch_size_time_axis+self.n_stack-1, axis=1), replay_buffer_state.experience)
        batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1).astype(jnp.float32), batch) 
        observations = batch.o_t
        terminations = batch.d_t
        truncations = batch.t_t
        resets = jnp.logical_or(terminations, truncations)

        def prepare_slices(i):
            return (
                jax.lax.dynamic_slice_in_dim(observations, i-self.n_stack+1, self.n_stack, axis=0), 
                jax.lax.dynamic_slice_in_dim(resets, i-self.n_stack+1, self.n_stack-1, axis=0)
            )
        
        def stack_observations(i):
            slices = prepare_slices(i)
            return self._stack_observations(observations_slice=slices[0], resets_slice=slices[1])

        stacked_observations = jax.vmap(stack_observations, in_axes=(0,), out_axes=0)(jnp.arange(self.n_stack-1, self.reanalyze_batch_size_time_axis+self.n_stack-1))
        chex.assert_shape(stacked_observations, (self.reanalyze_batch_size_time_axis, self.num_envs, *stacked_observations.shape[2:]))
    
        # Run MCTS for each stacked observation
        mcts_kwargs = self.act_config_reanalyze | {"temperature": 1.0}
        _, policy_output, _ = self._act(
            rng_key, 
            params, 
            stacked_observations.reshape(-1, *stacked_observations.shape[2:]), 
            **mcts_kwargs
        )

        # Get value targets
        if self.learn_config['loss']['value']['bootstrap_from'] == 'predicted':
            new_values = policy_output['predicted_value']
        elif self.learn_config['loss']['value']['bootstrap_from'] == 'mcts':
            new_values = policy_output["value"]
        else:
            raise ValueError(f"Invalid bootstrap from: {self.learn_config['loss']['value']['bootstrap_from']}")

        # reanalyze_logs = {"reanalyze_"+key: value for key, value in act_logs.items()}
        new_values = policy_output['value']
        new_policies = policy_output['policy']
        new_values = new_values.reshape(self.reanalyze_batch_size_time_axis, self.num_envs, *new_values.shape[1:])
        new_policies = new_policies.reshape(self.reanalyze_batch_size_time_axis, self.num_envs, *new_policies.shape[1:])
        new_v_t = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), new_values) # [num_timesteps, num_envs, ...] -> [num_envs, num_timesteps, ...]
        new_pi_t = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), new_policies) # [num_timesteps, num_envs, ...] -> [num_envs, num_timesteps, ...]
        chex.assert_axis_dimension(new_v_t, axis=1, expected=self.reanalyze_batch_size_time_axis)
        chex.assert_axis_dimension(new_pi_t, axis=1, expected=self.reanalyze_batch_size_time_axis)

        # Instead of dynamic slicing, use lax.dynamic_slice/dynamic_update_slice
        new_v_t = jax.lax.dynamic_update_slice_in_dim(
            replay_buffer_state.experience.v_t,
            new_v_t,
            start_index=minibatch_start_index*self.reanalyze_batch_size_time_axis + self.n_stack - 1,  # start indices for each dimension
            axis=1
        )
        new_pi_t = jax.lax.dynamic_update_slice_in_dim(
            replay_buffer_state.experience.pi_t,
            new_pi_t,
            start_index=minibatch_start_index*self.reanalyze_batch_size_time_axis + self.n_stack - 1,  # start indices for each dimension
            axis=1
        )

        # Average the logs
        # reanalyze_logs = jax.tree.map(lambda x: jnp.mean(jnp.array(x), axis=0), reanalyze_logs)
        reanalyze_logs = {}

        # Update the replay buffer state
        return replay_buffer_state.replace(
            experience=replay_buffer_state.experience._replace(
                v_t=new_v_t,
                pi_t=new_pi_t
            )
        ), reanalyze_logs
    
    # https://github.com/instadeepai/flashbax/issues/52
    @functools.partial(jax.jit, static_argnums=(0), donate_argnums=(1,))
    def _recompute_sum_tree(self, replay_buffer_state):
        sum_tree_state = replay_buffer_state.sum_tree_state
        leaf_start_idx = 2**sum_tree_state.tree_depth - 1
        leaf_end_idx = 2**(sum_tree_state.tree_depth + 1) - 1
        new_nodes = jnp.zeros_like(sum_tree_state.nodes)
        new_nodes = new_nodes.at[leaf_start_idx:leaf_end_idx].set(sum_tree_state.nodes[leaf_start_idx:leaf_end_idx])
        def compute_level(i, nodes):
            level_idx = sum_tree_state.tree_depth - 1 - i
            start_idx = (1 << level_idx) - 1
            size = 1 << level_idx
            def update_parent(j, nodes):
                parent_idx = start_idx + j
                left_child = 2 * parent_idx + 1
                right_child = 2 * parent_idx + 2
                nodes = nodes.at[parent_idx].set(
                    nodes[left_child] + nodes[right_child]
                )
                return nodes
            nodes = jax.lax.fori_loop(0, size, update_parent, nodes)
            return nodes
        new_nodes = jax.lax.fori_loop(
            0, sum_tree_state.tree_depth,
            compute_level,
            new_nodes
        )
        replay_buffer_state = replay_buffer_state.replace(sum_tree_state=replay_buffer_state.sum_tree_state.replace(nodes=new_nodes))
        return replay_buffer_state

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(3,))
    def _reanalyze(self, rng_key, params, replay_buffer_state):
        """Reanalyze the replay buffer."""

        replay_buffer_current_index = replay_buffer_state.running_index - (self.n_stack-1)

        # Divide the replay buffer into minibatches
        # The size of the minibatches is decided by the memory bandwidth of the GPU
        # ~2000 for 2080ti and ~5000 for V100,A40,A100
        # Use larger minibatches can speed up reanalyze
        def cond_fn(state):
            i_minibatch, rng_key, buffer_state = state
            num_minibatches = jnp.ceil(replay_buffer_current_index / self.reanalyze_batch_size_time_axis)
            return i_minibatch < num_minibatches
        
        def body_fn(state):
            i_minibatch, rng_key, buffer_state = state
            rng_key, subkey = jax.random.split(rng_key)
            buffer_state, _ = self._reanalyze_batch(subkey, params, buffer_state, i_minibatch)
            return i_minibatch + 1, rng_key, buffer_state
        
        init_state = (0, rng_key, replay_buffer_state)
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        final_replay_buffer_state = final_state[2]
        
        reanalyze_logs = {}
        return final_replay_buffer_state, reanalyze_logs
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _log_replay_distribution(self, rng_key, replay_buffer_state):
        """Log the replay distribution by taking a sample from the replay buffer."""
        sampled_indices = []
        for i in range(self.learn_config['replay_buffer']['log']['sample_size']):
            rng_key, subkey = jax.random.split(rng_key)
            sample = self.replay_buffer.sample(replay_buffer_state, subkey)
            indices = sample.indices
            sampled_indices.append(indices)
        concatenated_indices = jnp.concatenate(sampled_indices)
        length = (self.replay_buffer_max_size//self.num_envs)+self.n_stack-1
        i = concatenated_indices // length
        j = concatenated_indices % length
        converted_indices = j * self.num_envs + i
        return converted_indices
    
    def learn(self, rng_key: chex.PRNGKey, agent_state: AgentState, logs: dict = {"Training": {}, "Profiling": {}}):
        """Update agent parameters using experiences from replay buffer."""

        if self.replay_buffer.can_sample(agent_state.replay_buffer_state):
            
            rng_key, subkey = jax.random.split(rng_key)
            beta = self._beta_fn(agent_state.learn_step)

            start = time.time()
            # From time to time, we log the replay distribution for debugging
            if agent_state.learn_step % self.learn_config['replay_buffer']['log']['per_learn_step'] == 0:
                converted_indices = self._log_replay_distribution(subkey, agent_state.replay_buffer_state)
                logs['Training']['replay_distribution'] = converted_indices
            
            # Profile log replay distribution
            if self.profile:
                jax.tree.map(jax.block_until_ready, agent_state.replay_buffer_state)
                logs['Profiling']['log_replay_distribution'] = time.time() - start
            
            start = time.time()
            # Learn
            new_params, new_opt_state, new_replay_buffer_state, training_logs = self._learn(
                subkey, 
                agent_state.params, 
                agent_state.opt_state,
                agent_state.replay_buffer_state,
                beta
            )
            
            # Update counters
            new_learn_step = agent_state.learn_step + 1
            env_step = agent_state.env_step
            new_gradient_step = agent_state.gradient_step + self.sgd_steps
            logs['Training'].update(training_logs)
            logs['Training']['beta'] = beta
            logs['Training']['gradient_step'] = new_gradient_step
            logs['Training']['lr'] = self.lr_schedule(agent_state.gradient_step)
            
            # Profile SGD steps
            if self.profile:
                jax.tree.map(jax.block_until_ready, new_params)
                logs['Profiling']['sgd'] = time.time() - start

            # Recompute sum tree
            start = time.time()
            if new_learn_step % self.learn_config['replay_buffer']['PER']['correction_per_learn_step'] == 0:
                new_replay_buffer_state = self._recompute_sum_tree(new_replay_buffer_state)

            # Profile recompute sum tree
            if self.profile:
                jax.tree.map(jax.block_until_ready, new_replay_buffer_state)
                logs['Profiling']['recompute_sum_tree'] = time.time() - start
            
            # Reanalyze
            start = time.time()
            if self.reanalyze_per_learn_step is not None and new_learn_step % self.reanalyze_per_learn_step == 0:
                rng_key, subkey = jax.random.split(rng_key)
                new_replay_buffer_state, reanalyze_logs = self._reanalyze(subkey, new_params, new_replay_buffer_state)
                new_reanalyze_step = agent_state.reanalyze_step + 1
                logs['Training'].update(reanalyze_logs)
                print("Reanalyzed replay buffer.")
            else:
                new_reanalyze_step = agent_state.reanalyze_step
                new_replay_buffer_state = new_replay_buffer_state
            logs['Training']['reanalyze_step'] = new_reanalyze_step
            
            # Profile reanalyze
            if self.profile:
                jax.tree.map(jax.block_until_ready, new_replay_buffer_state)
                logs['Profiling']['reanalyze'] = time.time() - start
            
            # Return new agent state
            return self.AgentState(
                env_step=env_step,
                learn_step=new_learn_step,
                gradient_step=new_gradient_step,
                reanalyze_step=new_reanalyze_step,
                params=new_params,
                opt_state=new_opt_state,
                replay_buffer_state=new_replay_buffer_state
            )
        else:
            return agent_state

    @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(3,))
    def _randomize_action_if_replay_buffer_is_small(
        self,
        rng_key: chex.PRNGKey,
        replay_buffer_state: fbx.trajectory_buffer.BufferState,
        action: chex.Array
    ):
        """Randomize action."""
        rng_key, subkey = jax.random.split(rng_key)
        random_action = jax.random.randint(
            subkey, 
            shape=action.shape, 
            minval=0, 
            maxval=self.num_actions
        )
        return jnp.where(self.replay_buffer.can_sample(replay_buffer_state), action, random_action)