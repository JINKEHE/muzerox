"""
Atari environment wrappers and utilities for reinforcement learning.
This module provides wrappers for Atari environments following common preprocessing
steps used in research papers like MuZero and EfficientZero.
Furthermore, it provides a Jax-compatible wrapper with explicit memory management to avoid memory fragmentation.

Adapted from
1. https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/atari_preprocessing.py
2. https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/stateful_observation.py
3. https://github.com/YeWR/EfficientZero/blob/468bb0309f6d5a632a53da9c7d329f88fc9ebf8e/core/utils.py
"""

import gymnasium as gym
import ale_py
import cv2
from typing import Any
import functools
import jax
import numpy as np

# Register Atari environments
gym.register_envs(ale_py)

# Suppress ALE welcome messages
ale_py.ALEInterface.setLoggerMode(ale_py.LoggerMode.Error)

class AtariWrapper(gym.Wrapper):
    """
    Wrapper for Atari 2600 preprocessing.
    """
    def __init__(
        self, 
        env: gym.Env, 
        *,
        noop_max: int = 30,
        n_skip: int = 4,
        n_stack: int = 4,
        screen_size: int = 96,
        terminal_on_life_loss: bool = False,
        max_episode_steps: int = 3000,
        clip_reward: bool = True
    ):
        gym.Wrapper.__init__(self, env)

        """Check args"""
        # Screen size
        assert isinstance(screen_size, int) and screen_size > 0
        self.screen_size = (screen_size, screen_size)
        self.screen_shape = (screen_size, screen_size, 3)
        self.original_screen_shape = env.observation_space.shape
        # Frame skip
        self.n_frame_skip = n_skip
        assert n_skip >= 0
        if n_skip > 1 and getattr(env.unwrapped, "_frameskip", None) != 1:
            raise ValueError(
                "Disable frame-skipping in the original env. Otherwise, more than one frame-skip will happen as through this wrapper"
            )
        # Observation stack
        self.n_stack = n_stack
        self.observation_shape = (screen_size, screen_size, 3*n_stack)
        assert n_stack >= 0
        # Noop max
        assert noop_max >= 0
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
        # Terminal on life loss
        self.terminal_on_life_loss = terminal_on_life_loss
        self.hard_reset = True
        # Max episode steps
        self.max_episode_steps = max_episode_steps
        assert max_episode_steps > 0
        # Clip reward
        self.clip_reward = clip_reward

        # Set up observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8
        )

        # Set up action space
        self.action_space = gym.spaces.Discrete(env.action_space.n)

        # Initialize variables
        self.episode_steps = 0
        self.total_reward = 0.0 # Sum of potentially clipped rewards with terminal on life loss
        self.lives = 0
        self.game_over = False

        # Set up frame buffer
        self.frame_buffer = [
            np.empty(self.original_screen_shape, dtype=np.uint8),
            np.empty(self.original_screen_shape, dtype=np.uint8),
        ]

        # Set up placeholder for stacked observations
        self.stacked_obs = np.empty(self.observation_shape, dtype=np.uint8)

        self.hard_reset = True

    @property
    def ale(self):
        """Make ale as a class property to avoid serialization error."""
        return self.env.unwrapped.ale
    
    def step(self, action):
        """
        Step the environment with the given action.

        Relevant wrappers:
        - FrameSkip
        - ObservationStack
        - ClipReward
        - TerminalOnLifeLoss
        - TimeLimit
        """
        total_reward, terminated, truncated, info = 0.0, False, False, {}

        for t in range(self.n_frame_skip):
            frame, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            
            # Only get screen at the last two frames
            if t == self.n_frame_skip - 2:
                self.ale.getScreenRGB(self.frame_buffer[1])
            elif t == self.n_frame_skip - 1:
                self.ale.getScreenRGB(self.frame_buffer[0])

        # Observation stacking
        current_obs = self._get_obs()
        self.stacked_obs[:,:,:-3] = self.stacked_obs[:,:,3:]
        self.stacked_obs[:,:, -3:] = current_obs

        # Maximum episode steps
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
        
        # Clip reward
        if self.clip_reward:
            reward = np.sign(total_reward)
        else:
            reward = total_reward
        self.total_reward += reward
        
        # Terminal on life loss
        if self.terminal_on_life_loss:
            if terminated:
                self.hard_reset = True
            else:
                if self.ale.lives() < self.lives and self.ale.lives() > 0:
                    terminated = True
                    self.hard_reset = False
        else:
            # If life loss and termination_on_life_loss is not turned on, duplicate the current observation to avoid mismatch with training
            if self.ale.lives() < self.lives and self.ale.lives() > 0:
                self.stacked_obs = np.tile(current_obs, (1, 1, self.n_stack))

        self.lives = self.ale.lives()

        return self.stacked_obs.astype(np.uint8), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.

        Relevant wrappers:
        - TimeLimit
        - NoopReset
        - ObservationStack
        - TerminalOnLifeLoss
        - FrameSkip
        """

        if self.terminal_on_life_loss is True and self.hard_reset is False:
            # Keep obs buffer if soft reset
            reset_info = {}
        else:
            # Hard reset
            _, reset_info = self.env.reset(seed=seed, options=options)

            # Noop reset
            noops = (
                self.env.unwrapped.np_random.integers(1, self.noop_max + 1)
                if self.noop_max > 0
                else 0
            )
            for _ in range(noops):
                _, _, terminated, truncated, step_info = self.env.step(0)
                reset_info.update(step_info)
                if terminated or truncated:
                    _, reset_info = self.env.reset(seed=seed, options=options)
                    
            # Obs buffer
            self.ale.getScreenRGB(self.frame_buffer[0])
            self.frame_buffer[1].fill(0)
        
            # Observation stacking
            current_obs = self._get_obs()
            self.stacked_obs = np.tile(current_obs, (1, 1, self.n_stack))
        
        reset_info['total_reward'] = self.total_reward
        reset_info['episode_steps'] = self.episode_steps

        # Maximum episode steps
        self.episode_steps = 0

        # Total reward
        self.total_reward = 0.0

        # Terminal on life loss
        self.lives = self.ale.lives()
        
        return self.stacked_obs.astype(np.uint8), reset_info
    
    def _get_obs(self):
        if self.n_frame_skip > 1:
            np.maximum(self.frame_buffer[0], self.frame_buffer[1], out=self.frame_buffer[0])
        return cv2.resize(
            self.frame_buffer[0],
            self.screen_size,
            interpolation=cv2.INTER_AREA,
        )

class JaxVectorizedWrapper(gym.vector.VectorWrapper):
    
    def __init__(self, env: gym.vector.VectorEnv, **kwargs):
        super().__init__(env, **kwargs)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None
    ):
        obs, info = super().reset(seed=seed, options=options)
        self.last_stacked_obs = jax.device_put(obs)
        return self.last_stacked_obs, info
    
    def step(self, action):
        action = jax.device_get(action)
        obs, reward, terminated, truncated, info = super().step(action)
        self.last_stacked_obs = jax.device_put(obs)
        return self.last_stacked_obs, jax.device_put(reward), jax.device_put(terminated), jax.device_put(truncated), info

def make_atari(
    game_name: str,
    noop_max: int = 30,
    n_skip: int = 4,
    n_stack: int = 4,
    screen_size: int = 96,
    terminal_on_life_loss: bool = True,
    max_episode_steps: int = 3000,  # 3000 for training, 27000 for final test
    clip_reward: bool = True,
    env_kwargs: dict = None,
) -> gym.Env:
    """Create and configure a single Atari environment with standard wrappers.
    
    Args:
        game_name: Name of the Atari game
        noop_max: Maximum number of random no-op actions at start
        n_skip: Number of frames to skip (action repeat)
        n_stack: Number of observations to stack
        screen_size: Size to resize frames to
        terminal_on_life_loss: End episode on life loss
        max_episode_steps: Maximum steps per episode
        clip_reward: Whether to clip rewards to {-1, 0, 1}
        env_kwargs: Additional kwargs to pass to gym.make()
        
    Returns:
        Wrapped Atari environment
    """
    env_kwargs = env_kwargs or {}
    env_id = f"{game_name}NoFrameskip-v4"
    env = gym.make(env_id, **env_kwargs)
    
    env = AtariWrapper(
        env, 
        noop_max=noop_max, 
        n_skip=n_skip, 
        n_stack=n_stack, 
        screen_size=screen_size, 
        terminal_on_life_loss=terminal_on_life_loss, 
        max_episode_steps=max_episode_steps, 
        clip_reward=clip_reward)
    
    return env

def make_vectorized_atari(
    game_name: str,
    noop_max: int = 30,
    n_skip: int = 4,
    n_stack: int = 4,
    screen_size: int = 96,
    terminal_on_life_loss: bool = True,
    max_episode_steps: int = 3000,
    clip_reward: bool = True,
    env_kwargs: dict = None,
    num_envs: int = 8,
    vectorization_mode: str = "async",
    jaxify: bool = True
) -> gym.vector.VectorEnv:
    """Create multiple vectorized Atari environments.
    
    Args:
        Similar to make_atari() plus:
        num_envs: Number of environments to create
        vectorization_mode: "async" or "sync" vectorization
        
    Returns:
        Vectorized Atari environments
    """
    env_kwargs = env_kwargs or {}
    env_id = f"{game_name}NoFrameskip-v4"
    
    wrappers = [
        functools.partial(
            AtariWrapper, 
            noop_max=noop_max,
            n_skip=n_skip,
            n_stack=n_stack,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            max_episode_steps=max_episode_steps,
            clip_reward=clip_reward
        )
    ]

    envs = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,
        wrappers=wrappers,
        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
        **env_kwargs
    )

    if jaxify:
        envs = JaxVectorizedWrapper(envs)

    return envs