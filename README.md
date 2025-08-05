# MuZeroX

MuZeroX is a JAX implementation of of the MuZero algorithm, optimized for Atari games.


## Key Libraries
- **[mctx](https://github.com/google-deepmind/mctx)** - GPU‑accelerated Monte‑Carlo Tree Search in JAX
    - We slightly modified mctx to implement MuZero’s original min–max normalization within the MCTS, which we found improves the performance.
- **[ALE](https://github.com/mgbellemare/Arcade-Learning-Environment)** & **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)** – Atari environments.  
- **[Flashbax](https://github.com/google/flashbax)** – GPU-accelerated (prioritized) experience replay.
- **[flax.linen](https://github.com/google/flax)** – Neural network in JAX.
- **[TensorboardX](https://github.com/lanpa/tensorboardX)** & **[Weights & Biases (wandb)](https://wandb.ai/)** – Experiment management.
- **[OmegaConf](https://github.com/omry/omegaconf)** – Experiment configuration.

## Feature
With [Flashbax](https://github.com/google/flashbax), all replay buffers reside on GPU, avoiding data transfer between RAM and VRAM.

With [mctx](https://github.com/google-deepmind/mctx), all tree search runs on GPU, again avoiding RAM‑VRAM traffic.

Combined, these yield a GPU‑native MuZero pipeline that’s faster in throughput.

## Installation
### Conda
```bash
conda env create -f conda/env.yml
conda activate muzerox
```
### Apptainer (container)
```bash
sudo ./apptainer/build
./apptainer/run <command>
```

## Run MuZero in the Atari-100K setting
```bash
python3 -m src.main experiments/100K environments/100K.yaml env.common.game_name=Breakout
```
Replace ```Breakout``` with any supported Atari game. Create a new yaml file or add arguments at the end of the command to change hyperparameters.

## Acknowledgements
We are inspired by the following codebases:
* [jax_muzero](https://github.com/Hwhitetooth/jax_muzero)
* [EfficientZero](https://github.com/YeWR/EfficientZero)
* [muzero-general](https://github.com/werner-duvaud/muzero-general)