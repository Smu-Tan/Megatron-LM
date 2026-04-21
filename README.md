Attention Residuals for Megatron-LM
===================================

This branch is a research implementation of **Attention Residuals** for
Megatron Core, built to reproduce and stress-test the Full and Block variants
from [`arXiv:2603.15031`](https://arxiv.org/abs/2603.15031) inside a real
Megatron-LM training stack.

The short version: fixed residual accumulation is replaced by a learned
depth-wise attention over previous residual-producing hidden states. In this
implementation, attention and MLP are treated as separate residual-producing
sublayers, and a final AttnRes aggregation is applied before the transformer
block's final layer norm.

This is intentionally labeled as a **research prototype**. It is meant for
reproduction, ablations, and systems experiments. It is not yet a claim that
every Megatron model family, inference path, and parallelism composition is
production-ready.

## What Is In This Branch

- **Full AttnRes**: attends over every previous residual-producing hidden state.
- **Block AttnRes**: compresses depth into block summaries and attends over
  completed blocks plus the current partial block.
- **Final block aggregation**: the final hidden state is also produced through
  AttnRes, rather than falling back to the last MLP output.
- **Multiple implementations**:
  - `torch`: reference implementation
  - `checkpointed`: recompute AttnRes internals in backward to reduce memory
  - `triton`: Triton forward kernels
  - `triton_bwd`: Triton forward and backward recomputation kernels
- **Megatron integration** through config flags, transformer block/layer hooks,
  example launchers, and unit tests.

## Quick Start

Use Block AttnRes with the optimized Triton backward path for the first serious
run:

```bash
--attention-residuals \
--attention-residual-type block \
--attention-residual-num-blocks 8 \
--attention-residual-implementation triton_bwd
```

Example scripts live in [`examples/attention_residuals/`](examples/attention_residuals/).
They assume you already have a tokenizer and a Megatron indexed dataset prefix.

```bash
export TOKENIZER_MODEL=/path/to/llama3-tokenizer
export DATA_PREFIX=/path/to/megatron_text_document

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

## Code Guide

- Core module:
  [`megatron/core/transformer/attention_residuals.py`](megatron/core/transformer/attention_residuals.py)
- Config flags:
  [`megatron/core/transformer/transformer_config.py`](megatron/core/transformer/transformer_config.py)
- Transformer block integration:
  [`megatron/core/transformer/transformer_block.py`](megatron/core/transformer/transformer_block.py)
- Transformer layer integration:
  [`megatron/core/transformer/transformer_layer.py`](megatron/core/transformer/transformer_layer.py)
- Tests:
  [`tests/unit_tests/transformer/test_attention_residuals.py`](tests/unit_tests/transformer/test_attention_residuals.py)
- Feature documentation:
  [`docs/user-guide/features/attention_residuals.md`](docs/user-guide/features/attention_residuals.md)
- Reproduction launchers:
  [`examples/attention_residuals/`](examples/attention_residuals/)

## Reproduction Notes

The paper notes that Full AttnRes has no extra activation-memory overhead in
vanilla training because the needed activations are already retained. Megatron
training usually uses selective recomputation, FP8/Transformer Engine kernels,
and multiple parallelism dimensions, so the naive implementation can become
memory- and speed-sensitive. This branch keeps Full AttnRes as the semantic
reference and uses checkpointing/Triton kernels plus Block AttnRes to study that
systems gap.

Paper alignment checklist:

| Paper idea | This branch |
| --- | --- |
| Self-attention and MLP are separate residual-producing sublayers | implemented in `transformer_layer.py` |
| Full AttnRes attends over all previous residual states | `ATTENTION_RESIDUAL_TYPE=full` |
| Block AttnRes reduces depth candidates through block summaries | `ATTENTION_RESIDUAL_TYPE=block` |
| Final layer output should also be AttnRes-produced | implemented as `final_attn_res` in `transformer_block.py` |
| Full AttnRes overhead depends on whether activations are already retained | measured with Megatron recomputation/FP8/parallelism settings |

Observed development runs:

| Setup | Mode | Approx. speed | Notes |
| --- | --- | ---: | --- |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | baseline | 1040 ms/iter | matched Llama-style FP8 run |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Full, checkpointed | 4315 ms/iter | memory-safe but slow |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Full, Triton forward | 3922 ms/iter | forward kernel only |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Full, Triton forward/backward | 1690 ms/iter | optimized Full reference |
| 16 layers, seq 1024, TP=2, mbs=8, gbs=32 | Block, Triton forward/backward | faster than Full | short runs track Full-like loss |
| 32 layers, seq 4096, 4x H100 NVL, TP=2, CP=2, mbs=8, gbs=32 | baseline | 3695 ms/iter | 427.9 TFLOP/s/GPU at step 480 |

These numbers are development measurements, not official Megatron benchmarks.
They are included to make the implementation tradeoffs visible. A public W&B
project link can be added here once the final reproduction runs are curated:

```text
W&B project: TBD
```

## Current Scope

Exercised:

- Dense GPT/Llama-style decoder training.
- Transformer Engine FP8 training.
- Tensor parallelism, context parallelism, and sequence parallelism.
- Full and Block AttnRes with `torch`, `checkpointed`, `triton`, and
  `triton_bwd` implementations.

Not yet supported or not yet validated:

- MoE MLP layers.
- Cross-attention layers.
- Pipeline parallelism greater than one stage.
- Full-layer activation recomputation with AttnRes.
- Inference/KV-cache paths.

## Upstream Megatron-LM README

The original Megatron-LM README is kept below for installation, upstream
documentation, and general project context.

<details>
<summary>Show upstream README</summary>

<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts. Best for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (e.g. `MAX_JOBS=4 uv pip install -e .`).

For NGC container setup and all installation options, see the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core

# Latest News

- **[2026/03]** **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.17.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with MCore.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Previous News</summary>

- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

</details>

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Roadmaps

- **[MoE Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - DeepSeek-V3, Qwen3, advanced parallelism, FP8 optimizations, and Blackwell enhancements

# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, we appreciate that you use the following citations:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```

</details>
