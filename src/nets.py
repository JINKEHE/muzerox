"""Neural network modules in Flax.Linen, adapted from EfficientZero."""
import jax
import jax.numpy as jnp
from flax import linen as nn

activation_fn = nn.leaky_relu
initializer_fn = nn.initializers.variance_scaling(scale=1.0, mode="fan_out", distribution="normal")
norm_fn = lambda: nn.LayerNorm(reduction_axes=(-1), feature_axes=(-1))
prediction_net_norm_fn = lambda: nn.LayerNorm(reduction_axes=(-3, -2, -1), feature_axes=(-3, -2, -1))

def scale_gradient(x, scale):
    """https://arxiv.org/src/1911.08265v2/anc/pseudocode.py."""
    return x * scale + (1 - scale) * jax.lax.stop_gradient(x)

def normalize_state(x):
    """https://arxiv.org/pdf/1911.08265, Appendix G."""
    min_val = jnp.min(x, axis=-1, keepdims=True)
    max_val = jnp.max(x, axis=-1, keepdims=True)
    return (x - min_val) / jnp.clip(max_val - min_val, a_min=1e-6)

class ResidualBlock(nn.Module):
    """A residual block."""
    features: int = 64
    stride: int = 1
    downsample: bool = False

    def setup(self):
        self.conv1 = nn.Conv(features=self.features, kernel_size=(3, 3), strides=self.stride, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.conv2 = nn.Conv(features=self.features, kernel_size=(3, 3), strides=1, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.norm1 = norm_fn()
        self.norm2 = norm_fn()
        if self.downsample:
            self.conv3 = nn.Conv(features=self.features, kernel_size=(3, 3), strides=2, padding="SAME", use_bias=False, kernel_init=initializer_fn)
    
    def __call__(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = activation_fn(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample:
            identity = self.conv3(identity)
        out = out + identity
        out = activation_fn(out)
        return out
        
class RepresentationNet(nn.Module):
    """Representation network."""
    features: int = 64
    normalize_state: bool = True
    
    def setup(self):
        self.conv1 = nn.Conv(features=self.features//2, kernel_size=(3,3), strides=2, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.norm1 = norm_fn()
        self.residual_block1 = ResidualBlock(features=self.features//2, stride=1, downsample=False)
        self.residual_block2 = ResidualBlock(features=self.features, stride=2, downsample=True)
        self.residual_block3 = ResidualBlock(features=self.features, stride=1, downsample=False)
        self.residual_block4 = ResidualBlock(features=self.features, stride=1, downsample=False)
        self.residual_block5 = ResidualBlock(features=self.features, stride=1, downsample=False)
    
    def __call__(self, x):
        out = self.conv1(x / 255.0)
        out = self.norm1(out)
        out = activation_fn(out)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = nn.avg_pool(out, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        out = self.residual_block4(out)
        out = nn.avg_pool(out, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        out = self.residual_block5(out)
        if self.normalize_state:
            out = normalize_state(out)
        return out

class DynamicsNet(nn.Module):
    """Dynamics network."""
    features: int
    normalize_state: bool = True

    def setup(self):
        self.conv_state = nn.Conv(features=self.features, kernel_size=(3,3), strides=1, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.residual_block_state = ResidualBlock(features=self.features, stride=1, downsample=False)

    def __call__(self, x):
        # https://arxiv.org/src/1911.08265v2/anc/pseudocode.py
        x = scale_gradient(x, 0.5)
        state = x[:,:,:,:-1]
        out = self.conv_state(x)
        out += state
        out = activation_fn(out)
        out = self.residual_block_state(out)
        next_state = out
        # https://arxiv.org/pdf/1911.08265, Appendix G
        if self.normalize_state:
            next_state = normalize_state(next_state)
        return next_state

class TwoLayerMLP(nn.Module):
    """Two-layer MLP."""
    out_features: int
    hidden_layer_features: int = 32
    zero_initialize_last_layer: bool = True
    
    def setup(self):
        self.fc1 = nn.Dense(features=self.hidden_layer_features, use_bias=False)
        self.norm1 = norm_fn()
        # https://arxiv.org/pdf/2111.00210, Appendix A.1
        if self.zero_initialize_last_layer:
            self.fc2 = nn.Dense(features=self.out_features, use_bias=True, kernel_init=nn.initializers.zeros_init())
        else:
            self.fc2 = nn.Dense(features=self.out_features, use_bias=True)
    
    def __call__(self, x):
        out = self.fc1(x)
        out = self.norm1(out)
        out = activation_fn(out)
        out = self.fc2(out)
        return out

class PredictionNet(nn.Module):
    """Prediction network."""
    num_actions: int
    num_bins: int
    residual_block_features: 64
    conv_features: int = 16
    mlp_hidden_layer_features: int = 32
    mlp_zero_initialize_last_layer: bool = True

    def setup(self):
        self.residual_block = ResidualBlock(features=self.residual_block_features, stride=1, downsample=False)

        # reward prediction
        self.conv_1x1_reward = nn.Conv(features=self.conv_features, kernel_size=(1,1), strides=1, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.mlp_reward = TwoLayerMLP(hidden_layer_features=self.mlp_hidden_layer_features, out_features=self.num_bins, zero_initialize_last_layer=self.mlp_zero_initialize_last_layer)
        self.norm_reward = prediction_net_norm_fn()

        # value prediction
        self.conv_1x1_value = nn.Conv(features=self.conv_features, kernel_size=(1,1), strides=1, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.mlp_value = TwoLayerMLP(hidden_layer_features=self.mlp_hidden_layer_features, out_features=self.num_bins, zero_initialize_last_layer=self.mlp_zero_initialize_last_layer)
        self.norm_value = prediction_net_norm_fn()

        # policy prediction
        self.conv_1x1_policy = nn.Conv(features=self.conv_features, kernel_size=(1,1), strides=1, padding="SAME", use_bias=False, kernel_init=initializer_fn)
        self.mlp_policy = TwoLayerMLP(hidden_layer_features=self.mlp_hidden_layer_features, out_features=self.num_actions, zero_initialize_last_layer=self.mlp_zero_initialize_last_layer)
        self.norm_policy = prediction_net_norm_fn()

    def __call__(self, x):

        out = self.residual_block(x)

        # reward prediction
        out_reward_logits = self.conv_1x1_reward(x)
        out_reward_logits = self.norm_reward(out_reward_logits)
        out_reward_logits = activation_fn(out_reward_logits)
        out_reward_logits = jax.lax.collapse(out_reward_logits, start_dimension=1)
        out_reward_logits = self.mlp_reward(out_reward_logits)

        # value prediction
        out_value_logits = self.conv_1x1_value(out)
        out_value_logits = self.norm_value(out_value_logits)
        out_value_logits = activation_fn(out_value_logits)
        out_value_logits = jax.lax.collapse(out_value_logits, start_dimension=1)
        out_value_logits = self.mlp_value(out_value_logits)

        # policy prediction
        out_policy_logits = self.conv_1x1_policy(out)
        out_policy_logits = self.norm_policy(out_policy_logits)
        out_policy_logits = activation_fn(out_policy_logits)
        out_policy_logits = jax.lax.collapse(out_policy_logits, start_dimension=1)
        out_policy_logits = self.mlp_policy(out_policy_logits)

        return out_reward_logits, out_value_logits, out_policy_logits
    
class EfficientZeroNet(nn.Module):
    """EfficientZero network."""
    num_actions: int
    num_bins: int
    representation_net_num_channels: int = 64
    prediction_net_num_channels: int = 16
    prediction_net_mlp_num_features: int = 32
    prediction_net_mlp_zero_initialize_last_layer: bool = True
    normalize_state: bool = True

    def setup(self):
        self.representation_net = RepresentationNet(
            features=self.representation_net_num_channels,
            name="representation_net",
            normalize_state=self.normalize_state
        )
        self.dynamics_net = DynamicsNet(
            features=self.representation_net_num_channels,
            name="dynamics_net",
            normalize_state=self.normalize_state
        )
        self.prediction_net = PredictionNet(
            residual_block_features=self.representation_net_num_channels,
            conv_features=self.prediction_net_num_channels,
            mlp_hidden_layer_features=self.prediction_net_mlp_num_features,
            mlp_zero_initialize_last_layer=self.prediction_net_mlp_zero_initialize_last_layer,
            num_actions=self.num_actions,
            num_bins=self.num_bins,
            name="prediction_net"
        )
    
    def encode(self, state):
        encoded_state = self.representation_net(state)
        return encoded_state
    
    def predict(self, encoded_state):
        reward_logits, value_logits, policy_logits = self.prediction_net(encoded_state)
        return reward_logits, value_logits, policy_logits
    
    def step(self, encoded_state, action):
        action_onehot = action[:, None, None, None] * jnp.ones((encoded_state.shape[0], encoded_state.shape[1], encoded_state.shape[2], 1)) / self.num_actions
        encoded_state_action = jnp.concatenate([encoded_state, action_onehot], axis=-1)
        next_state = self.dynamics_net(encoded_state_action)
        return next_state
    
    def unroll(self, encoded_state, actions):
        def body_fn(carry, action):
            encoded_state = carry
            next_state = self.step(encoded_state, action)
            return next_state, next_state
        _, unrolled_states = jax.lax.scan(body_fn, encoded_state, actions)
        return unrolled_states

    def encode_and_unroll(self, state, actions):
        encoded_state = self.encode(state)
        unrolled_encoded_states = self.unroll(encoded_state, actions)
        unrolled_encoded_states = jnp.concatenate([encoded_state[None], unrolled_encoded_states], axis=0)
        return unrolled_encoded_states
    
    def encode_and_unroll_and_predict(self, state, actions):
        unrolled_encoded_states = self.encode_and_unroll(state, actions)
        rewards_logits, values_logits, policies_logits = jax.vmap(self.predict, in_axes=0, out_axes=0)(unrolled_encoded_states)
        return rewards_logits, values_logits, policies_logits
    
    def __call__(self, state, action):
        encoded_state = self.encode(state)
        next_encoded_state = self.step(encoded_state, action)
        reward, value, policy = self.predict(next_encoded_state)
        return next_encoded_state, reward, value, policy