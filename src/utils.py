import jax
import jax.numpy as jnp
import chex
import rlax

def safe_log(x: chex.Array, eps: float = 1e-8) -> chex.Array:
    eps = jnp.asarray(eps, dtype=x.dtype)
    return jnp.log(jnp.clip(x, eps, 1.0))

def compute_kl_divergence_with_probs_and_probs(
    target_probs: chex.Array,
    predicted_probs: chex.Array,
    eps: float = 1e-8,
) -> chex.Array:
    dtype = jnp.float32 if target_probs.dtype == jnp.float16 else target_probs.dtype
    p = target_probs.astype(dtype).reshape(-1, target_probs.shape[-1])
    q = predicted_probs.astype(dtype).reshape(-1, predicted_probs.shape[-1])
    log_p = safe_log(p, eps)
    log_q = safe_log(q, eps)
    kl = jnp.sum(p * (log_p - log_q), axis=-1)
    return kl.reshape(target_probs.shape[:-1])

def compute_kl_divergence_with_probs_and_logits(
    target_probs: chex.Array, 
    predicted_logits: chex.Array,
    epsilon: float = 1e-8
) -> chex.Array:
    predicted_probs = jax.nn.softmax(predicted_logits, axis=-1)
    return compute_kl_divergence_with_probs_and_probs(target_probs, predicted_probs, epsilon)

def compute_entropy_with_probs(probs: chex.Array, epsilon: float = 1e-8) -> chex.Array:
    dtype = jnp.float32 if probs.dtype == jnp.float16 else probs.dtype
    probs = probs.astype(dtype)
    return -jnp.sum(probs * safe_log(probs, epsilon), axis=-1)

# Exactly from rlax but added clipping to improve numerical stability
def transform_to_2hot(
    scalar: chex.Array,
    min_value: float,
    max_value: float,
    num_bins: int) -> chex.Array:
    scalar = jnp.clip(scalar, min_value, max_value)
    scalar_bin = (scalar - min_value) / (max_value - min_value) * (num_bins - 1)
    lower, upper = jnp.floor(scalar_bin), jnp.ceil(scalar_bin)
    lower_value = (lower / (num_bins - 1.0)) * (max_value - min_value) + min_value
    upper_value = (upper / (num_bins - 1.0)) * (max_value - min_value) + min_value
    p_lower = jnp.clip((upper_value - scalar) / (upper_value - lower_value + 1e-5), 0.0, 1.0) 
    p_upper = 1 - p_lower
    lower_one_hot = rlax._src.base.one_hot(
        lower, num_bins, dtype=scalar.dtype) * jnp.expand_dims(p_lower, -1)
    upper_one_hot = rlax._src.base.one_hot(
        upper, num_bins, dtype=scalar.dtype) * jnp.expand_dims(p_upper, -1)
    return lower_one_hot + upper_one_hot