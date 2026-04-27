# Attention Residuals Examples

This directory contains small, reproducible launch examples for the Megatron
Core Attention Residuals implementation. The goal is to make the fork easy to
inspect and reproduce without hiding important assumptions behind local cluster
scripts.

This is a **research implementation / reproduction prototype**. It is useful for
comparing baseline training, Full AttnRes, and Block AttnRes under matched
Megatron settings. It is not yet a production support statement for all
Megatron model families and parallelism combinations.

## What Was Added

The implementation follows the paper's residual-producing sublayer view:
self-attention and MLP are treated as two separate residual-producing sublayers
per transformer layer.

Code map:

- `megatron/core/transformer/attention_residuals.py`: AttnRes state,
  FullAttnRes module, checkpointed/Triton implementations.
- `megatron/core/transformer/transformer_config.py`: CLI/config flags.
- `megatron/core/transformer/transformer_block.py`: block-level AttnRes state
  creation and final aggregation.
- `megatron/core/transformer/transformer_layer.py`: layer integration before
  attention and MLP.
- `tests/unit_tests/transformer/test_attention_residuals.py`: unit tests for
  the operator and state grouping.

The current transformer stack uses AttnRes in three places:

- before self-attention in every transformer layer
- before MLP in every transformer layer
- once at the transformer block output before final layer norm

## Required Inputs

Set these paths before launching:

```bash
export TOKENIZER_MODEL=/path/to/llama3-tokenizer
export DATA_PREFIX=/path/to/megatron_text_document
```

`DATA_PREFIX` should point to a Megatron indexed dataset prefix, without the
`.bin` or `.idx` suffix. These examples do not download model weights or
datasets.

## Baseline

```bash
WANDB_PROJECT=attention-residuals \
WANDB_EXP_NAME=baseline_1000steps \
TOKENIZER_MODEL=$TOKENIZER_MODEL \
DATA_PREFIX=$DATA_PREFIX \
TRAIN_ITERS=1000 \
LR_DECAY_ITERS=10000 \
LR_WARMUP_ITERS=100 \
./examples/attention_residuals/train_llama3_wikitext.sh baseline
```

## Full AttnRes

Full AttnRes attends over all previous residual-producing hidden states. Use it
as the semantic reference; it is more expensive than Block AttnRes.

```bash
WANDB_PROJECT=attention-residuals \
WANDB_EXP_NAME=full_triton_bwd_1000steps \
TOKENIZER_MODEL=$TOKENIZER_MODEL \
DATA_PREFIX=$DATA_PREFIX \
TRAIN_ITERS=1000 \
LR_DECAY_ITERS=10000 \
LR_WARMUP_ITERS=100 \
ATTENTION_RESIDUAL_TYPE=full \
ATTENTION_RESIDUAL_IMPLEMENTATION=triton_bwd \
./examples/attention_residuals/train_llama3_wikitext.sh attnres
```

## Block AttnRes

Block AttnRes groups sublayer outputs into depth blocks and attends over block
summaries plus the current partial block. It is the recommended path for larger
experiments.

```bash
WANDB_PROJECT=attention-residuals \
WANDB_EXP_NAME=block_n8_triton_bwd_1000steps \
TOKENIZER_MODEL=$TOKENIZER_MODEL \
DATA_PREFIX=$DATA_PREFIX \
TRAIN_ITERS=1000 \
LR_DECAY_ITERS=10000 \
LR_WARMUP_ITERS=100 \
ATTENTION_RESIDUAL_TYPE=block \
ATTENTION_RESIDUAL_NUM_BLOCKS=8 \
ATTENTION_RESIDUAL_IMPLEMENTATION=triton_bwd \
./examples/attention_residuals/train_llama3_wikitext.sh attnres
```

For a stack with `L` transformer layers there are `2L` residual-producing
sublayers. With `NUM_LAYERS=16` and `ATTENTION_RESIDUAL_NUM_BLOCKS=8`, each
block covers four sublayer outputs.

## Common Knobs

```bash
NUM_LAYERS=16
SEQ_LENGTH=1024
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=32
TP_SIZE=2
PP_SIZE=1
CP_SIZE=1
```

Use `EXIT_DURATION_IN_MINS=30` for wall-clock-limited experiments. Leave it
unset for step-limited experiments controlled by `TRAIN_ITERS`.

For 4-GPU 32-layer/4K experiments, a stable first layout is:

```bash
GPUS_PER_NODE=4
NUM_LAYERS=32
SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=4096
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=32
TP_SIZE=2
CP_SIZE=2
PP_SIZE=1
```

Megatron-FSDP experiments can be launched with:

```bash
USE_MEGATRON_FSDP=1
DATA_PARALLEL_SHARDING_STRATEGY=optim
DATA_PARALLEL_SHARDING_STRATEGY=optim_grads
DATA_PARALLEL_SHARDING_STRATEGY=optim_grads_params
```

The strategies correspond roughly to ZeRO-1-like, ZeRO-2-like, and ZeRO-3-like
sharding. FSDP sharding acts over the data-parallel group, so choose a layout
with `DP_SIZE > 1` if you want it to reduce memory.

## Implementation Modes

```bash
ATTENTION_RESIDUAL_IMPLEMENTATION=torch
ATTENTION_RESIDUAL_IMPLEMENTATION=checkpointed
ATTENTION_RESIDUAL_IMPLEMENTATION=triton
ATTENTION_RESIDUAL_IMPLEMENTATION=triton_bwd
```

`torch` is the reference implementation. `checkpointed` reduces saved forward
intermediates by recomputing during backward. `triton` fuses the forward
reduction/accumulation path. `triton_bwd` also uses Triton for backward
recomputation and is the recommended mode when Triton is available.

## Observed Development Results

The following are development measurements, not official Megatron benchmarks.
They are included to show the expected direction of overheads and the reason for
keeping both Full and Block variants.

All runs used Transformer Engine FP8 training and matched model/data settings
within each comparison.

| Setup | Mode | Approx. ms/iter | Notes |
| --- | --- | ---: | --- |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | baseline | 1040 | reference run |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Full, checkpointed | 4315 | memory-safe but slow |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Full, Triton fwd | 3922 | modest speedup over checkpointed |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Full, Triton fwd+bwd | 1690 | much closer to baseline |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Block, Triton fwd+bwd | close to Full loss, faster than Full | recommended scaling path |
| 32 layers, seq 4096, 4x H100 NVL, TP=2, CP=2, mbs=8, gbs=32 | baseline | 3695 | development run reached 480 steps in 30 minutes |

Interpretation:

- Full AttnRes should be treated as the semantic reference.
- The optimized Triton backward is important for making Full AttnRes usable.
- Block AttnRes keeps quality close in short runs while reducing depth-attention
  candidate count, so it is the practical path for longer contexts and larger
  stacks.

## Limitations

Currently supported and exercised:

- Dense decoder-only GPT/Llama-style models.
- Transformer Engine FP8 examples.
- Tensor parallelism, context parallelism, and sequence parallelism.
- Full and Block AttnRes with PyTorch/checkpointed/Triton modes.

Not yet supported or not yet validated:

- MoE MLP layers.
- Cross-attention layers.
- Pipeline parallelism greater than one stage.
- Full-layer activation recomputation with AttnRes.
- Inference/KV-cache paths.
