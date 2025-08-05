# Python libraries
import logging
import time
import random
import datetime
import sys

# Hyperparameters
from omegaconf import OmegaConf

# Jax libraries
import chex
import jax
import numpy as np
import jax.numpy as jnp

# Wandb and tensorboardX
import wandb
from tensorboardX import SummaryWriter

# Make Atari environments, agents and utilities
from . import make_atari
from . import agents

def sample_seed(rng_key: chex.PRNGKey):
    return rng_key, int(jax.random.randint(rng_key, (), minval=0, maxval=jnp.uint32(2**32-1), dtype=jnp.uint32))

class Experiment:
    
    def __init__(self):
        pass

    def run(self):
        while self.steps['update_step'] < self.training_config['update_steps']:
            self.step()

    def cleanup(self):
        self.training_envs.close()
        self.evaluation_envs.close()
        if self.config['logging']['use_wandb']:
            wandb.finish()

    def setup(self, config):

        # Merge config
        config = OmegaConf.create(config)
        cli_config = OmegaConf.from_cli()
        self.config = OmegaConf.merge(config, cli_config)
        self.training_config = self.config['training']

        # Initialize jax
        platform = jax.default_backend()
        logging.warning("Running on %s", platform)

        # Seeding
        self.rng_key = jax.random.PRNGKey(self.training_config['seed'])
        self.rng_key, random_seed = sample_seed(self.rng_key)
        self.rng_key, np_random_seed = sample_seed(self.rng_key)
        random.seed(random_seed)
        np.random.seed(np_random_seed)
        self.training_config["random_seed"] = random_seed
        self.training_config["np_random_seed"] = np_random_seed

        # Make training envs
        training_env_config = OmegaConf.merge(self.config['env']['common'], self.config['env']['train'])
        self.training_envs = make_atari.make_vectorized_atari(**training_env_config)

        # Make evaluation envs
        eval_env_config = OmegaConf.merge(self.config['env']['common'], self.config['env']['eval'])
        assert eval_env_config['terminal_on_life_loss'] == False, "Evaluation envs should not terminate on life loss"
        self.evaluation_envs = make_atari.make_vectorized_atari(**eval_env_config)
        self.config['env']['common']['num_actions'] = int(self.training_envs.single_action_space.n)
        self.config['env']['common']['observation_shape'] = self.training_envs.single_observation_space.shape
        self.config['env']['eval']['seeds'] = []

        # Initialize counters
        assert self.training_config['env_steps'] % training_env_config['num_envs'] == 0, "env_steps must be divisible by num_envs"
        self.online_update_steps = self.training_config['env_steps'] // training_env_config['num_envs']
        self.offline_update_steps = self.training_config['offline_update_steps']
        self.training_config["update_steps"] = self.online_update_steps + self.offline_update_steps
        self.learn_per_update_step = self.training_config['learn_per_update_step']
        self.eval_per_update_step = self.training_config['eval_per_update_step']
        self.save_model_per_update_step = self.training_config['save_model_per_update_step']
        self.steps = {
            'env_step': 0,
            'learn_step': 0,
            'update_step': 0,
            'episode_step': 0,
            'evaluation_step': 0
        }

        # Set up wandb logging (optional)
        self.work_dir = "results/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.config['logging']['use_wandb']:
            wandb.init(
                **self.config['logging']['wandb'],
                sync_tensorboard=False, 
                config=OmegaConf.to_container(self.config, resolve=True),
            )
            # if wandb is used then save results in wandb folder
            self.work_dir = wandb.run.dir
            OmegaConf.save(self.config, f"{self.work_dir}/exp_config.yaml")   
        print(f"results are saved in {self.work_dir}") 
        
        # Set up tensorboard logging (optional)
        if self.config['logging']['use_tensorboard']:
            self.summary_writer = SummaryWriter(self.work_dir)
        
        # Set up profiling logs
        self.profile = self.config['logging']['profiling'] 
        self.accumulated_profiling_logs = {}
        self.start_experiment = time.time()

        # Reset training envs
        self.rng_key, train_env_seed = sample_seed(self.rng_key)
        self.stacked_obs, info = self.training_envs.reset(seed=train_env_seed)
        self.config['env']['train']['seed'] = train_env_seed
        
        # Initialize agent
        self.agent = agents.MuZeroAgent(config=self.config)
        self.rng_key, net_init_key = jax.random.split(self.rng_key)
        self.agent_state = self.agent.init_state(net_init_key, self.stacked_obs)

    def step(self):
        
        # Start the timer 
        start_step = time.time()
        self.steps['update_step'] += 1

        # Initialize the logs
        logs = {
            'Profiling': {},
            'Training': {},
            'Evaluation': {}
        }

        # If data collection is not done, act, simulate, add experience to replay buffer
        if self.steps['env_step'] < self.training_config['env_steps']:

            # 1. Acting - agent.act()
            start_acting = time.time()
            self.rng_key, subkey = jax.random.split(self.rng_key)
            action, policy_output = self.agent.act(subkey, self.agent_state, self.stacked_obs, logs=logs)
            
            # Profile acting
            if self.profile:
                jax.tree.map(jax.block_until_ready, policy_output)
                logs['Profiling']['acting'] = time.time() - start_acting
    
            # 2. Simulating - env.step()
            start_simulating = time.time()
            stacked_obs, reward, terminated, truncated, info = self.training_envs.step(action)

            # Log episode statistics
            done = np.logical_or(terminated, truncated)
            if done.any():
                logs['Training']['mean_episode_return'] = float((info['total_reward'] * done).sum() / done.sum())
                logs['Training']['mean_episode_length'] = float((info['episode_steps'] * done).sum() / done.sum())
                self.steps['episode_step'] += done.sum()
            self.steps['env_step'] += self.training_envs.num_envs

            # Profile simulating
            if self.profile:
                logs['Profiling']['simulating'] = time.time() - start_simulating

            # 3. Observing - agent.observe() - for agents with replay buffer, add new observation to replay buffer
            start_observing = time.time()
            self.agent_state = self.agent.observe(
                self.agent_state, 
                stacked_obs=self.stacked_obs,
                action=action,
                policy_output=policy_output,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                logs=logs
            )
            self.stacked_obs = stacked_obs

            # Profile observing
            if self.profile:
                jax.tree.map(jax.block_until_ready, self.agent_state.replay_buffer_state.experience)
                logs['Profiling']['observing'] = time.time() - start_observing

        # 4. When we have enough data, we can start training
        if (self.steps['update_step'] <= self.online_update_steps and self.steps['update_step'] % self.training_config['learn_per_update_step'] == 0) or self.steps['update_step'] > self.online_update_steps:
            start = time.time()
            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.agent_state = self.agent.learn(subkey, self.agent_state, logs=logs)
            self.steps['learn_step'] += 1
            
            # Profile training
            if self.profile:
                jax.tree.map(jax.block_until_ready, self.agent_state.params)
                logs['Profiling']['training'] = time.time() - start             

        # 5. Every self.eval_per_update_step steps we evaluate the agent
        if self.steps['update_step'] % self.eval_per_update_step == 0:
            start = time.time()
            num_eval_envs = self.evaluation_envs.num_envs
            self.rng_key, eval_env_seed = sample_seed(self.rng_key)
            self.config['env']['eval']['seeds'].append(eval_env_seed)
            state, _ = self.evaluation_envs.reset(seed=eval_env_seed)
            finished = np.zeros((num_eval_envs,))
            episode_infos = {"total_reward": np.zeros((num_eval_envs,)), "episode_steps": np.zeros((num_eval_envs,))}
            step = 0
            while np.sum(finished) < num_eval_envs:
                self.rng_key, subkey = jax.random.split(self.rng_key)
                action, policy_output = self.agent.eval_act(
                    subkey, 
                    self.agent_state, 
                    state,
                    logs=logs
                )
                state, _, terminated, truncated, info = self.evaluation_envs.step(action)
                step += 1
                dones = np.logical_or(terminated, truncated)
                if dones.any():
                    for i in range(num_eval_envs):
                        if dones[i]==1 and finished[i]==0:
                            finished[i] = True
                            episode_infos["total_reward"][i] = info['total_reward'][i]
                            episode_infos["episode_steps"][i] = info['episode_steps'][i]
            print(f"Evaluation finished after {step} steps.")
            logs['Evaluation']['mean_episode_return'] = float(np.mean(episode_infos['total_reward']))
            logs['Evaluation']['mean_episode_length'] = float(np.mean(episode_infos['episode_steps']))
            logs['Evaluation']['max_episode_return'] = float(np.max(episode_infos['total_reward']))
            logs['Evaluation']['min_episode_length'] = float(np.min(episode_infos['episode_steps']))
            logs['Evaluation']['max_episode_length'] = float(np.max(episode_infos['episode_steps']))
            logs['Evaluation']['min_episode_return'] = float(np.min(episode_infos['total_reward']))
            self.steps['evaluation_step'] += 1

            # Profile evaluation
            if self.profile:
                logs['Profiling']['evaluation'] = time.time() - start

        # 6. Save model
        if self.steps['update_step'] % self.training_config['save_model_per_update_step'] == 0:
            self.agent.save(self.agent_state, self.work_dir, f"agent_state_{self.steps['update_step']}.pkl")

        # 7. Log
        if self.profile:
            # Need to block Jax's async operations for proper profiling
            jax.tree.map(jax.block_until_ready, self.agent_state) 
            logs['Profiling']['full_step'] = time.time() - start_step
            logs['Profiling']['SPS'] = 0 if self.steps['env_step'] >= self.training_config['env_steps'] else int(self.stacked_obs.shape[0] / logs['Profiling']['full_step'])
            logs['Profiling']['UPS'] = 1 / logs['Profiling']['full_step']
        
            # Accumulate profiling logs
            for key in list(logs['Profiling'].keys()):
                self.accumulated_profiling_logs[key] = self.accumulated_profiling_logs.get(key, 0.0) + logs['Profiling'][key]
                logs['Profiling']["accumulated_" + key] = self.accumulated_profiling_logs[key]

        # 8. Write logs to tensorboard and/or wandb
        start_logging = time.time()
        logs['Charts'] = self.steps
        for log_category, log in logs.items():
            for log_key, log_value in log.items():
                # Write to tensorboard
                if self.config['logging']['use_tensorboard']:
                    if isinstance(log_value, float) or jnp.isscalar(log_value):
                        self.summary_writer.add_scalar(f'{log_category}/{log_key}', log_value, self.steps['update_step'])
                    else:
                        self.summary_writer.add_histogram(f'{log_category}/{log_key}', log_value, self.steps['update_step'])
                # Write to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({f'{log_category}/{log_key}': log_value}, step=self.steps['update_step'])
        logging_time = time.time() - start_logging

        # 9. Print out time elapsed, SPS, UPS, etc.
        if self.steps['update_step'] % self.training_config['log_per_update_step'] == 0:
            if self.profile:
                print(f"time elapsed: {round(time.time()-self.start_experiment, 2)} sec, SPS: {logs['Profiling']['SPS']}, env_step: {self.steps['env_step']}, learn_step: {self.steps['learn_step']}, UPS: {logs['Profiling']['UPS']}, update_step: {self.steps['update_step']}")
            else:
                print(f"time elapsed: {round(time.time()-self.start_experiment, 2)} sec, env_step: {self.steps['env_step']}, learn_step: {self.steps['learn_step']}, update_step: {self.steps['update_step']}")
            print(f"logging time: {logging_time} sec")

        return 
    
if __name__ == '__main__':
    config_path = sys.argv[1]
    experiment = Experiment()
    config = OmegaConf.load(config_path)
    experiment.setup(config)
    experiment.run()
    experiment.cleanup()