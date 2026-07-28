# Mini-LLM from Scratch with KV Caching — Neural Networks for Data Science (25/26)

A from-scratch implementation of a decoder-only Transformer language model in JAX/Equinox, trained on WikiText-2, with a custom Key-Value (KV) cache built to accelerate autoregressive inference. This project was completed as a project for the *Neural Networks for Data Science* course (2025/26).

## Overview

The assignment investigates the computational bottleneck of autoregressive text generation in Transformer-based language models and implements a solution used throughout modern LLM serving stacks: the **KV cache**. The project is organized in two complementary phases — first a baseline decoder-only Transformer trained and evaluated with standard full-sequence forward passes, then a functionally identical model augmented with a static, pre-allocated KV cache for single-token incremental decoding. The two implementations are verified for numerical equivalence and benchmarked for generation speed.

## Motivation

During autoregressive generation, a Transformer predicts one token at a time, conditioning on all previously generated tokens. Without caching, generating token $t$ requires recomputing the key and value projections for every position $1, \dots, t-1$, even though those projections do not change once computed. This yields $O(n^2)$ redundant attention computation per generated sequence of length $n$, and $O(n^3)$ total FLOPs.

Because Transformer self-attention is causal, the key and value vectors $K_j = W_K x_j$ and $V_j = W_V x_j$ for a past token $j$ are fixed functions of $x_j$ and the (frozen, post-training) weights — they never change as future tokens are appended. The KV cache exploits this invariance: each $(K_j, V_j)$ pair is computed once and reused for all subsequent decoding steps, reducing the incremental cost per step from $O(t)$ recomputation to $O(1)$ new projections plus an $O(t)$ attention read.

## Architecture

The base model is a standard **decoder-only Transformer language model** (GPT-2-style), consisting of:

- Learned token embeddings and learned absolute positional embeddings.
- $L$ stacked pre-norm Transformer blocks, each containing:
  - Multi-head causal self-attention.
  - A position-wise two-layer feed-forward network with GELU activation.
  - Residual connections around both sublayers.
- A final LayerNorm.
- A weight-tied output projection (the LM head reuses the token embedding matrix).

At layer level, for input $x$:

$$x \leftarrow x + \text{Attn}(\text{LN}_1(x))$$

$$x \leftarrow x + \text{FFN}(\text{LN}_2(x))$$

Self-attention is computed per head as:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V$$

where $d_h$ is the per-head dimension and a lower-triangular causal mask prevents attention to future positions.

## KV-Cache Design

The KV-cache variant reuses the exact same architecture and trained weights, differing only in the attention layer's execution mode:

- **Training mode** (`cache=None`): identical to the baseline — the full sequence is processed in parallel with a causal mask, used both for training and for equivalence verification.
- **Inference mode** (`cache` provided): a single token is processed at a time; its key and value vectors are written into a pre-allocated buffer of shape $(H, T_{\max}, d_h)$ via `jax.lax.dynamic_update_slice`, and attention is computed against all populated cache slots up to the current step.

Static, pre-allocated buffers (rather than dynamic concatenation) are used because JAX's JIT compilation requires fixed tensor shapes; growing tensors would force recompilation at every decoding step.

Generation follows the standard two-phase inference protocol:

1.  **Prefill**: the prompt is fed token-by-token to populate the cache for all prompt positions (equivalent in cost to one parallel forward pass).
2.  **Decode**: each newly generated token is fed individually; only its own key/value pair is computed, while all past keys/values are read from the cache.

### Memory footprint

For $L$ layers, $H$ heads, head dimension $d_h$, and maximum sequence length $T$:

$$\text{KV cache size} = 2 \cdot L \cdot H \cdot T \cdot d_h \cdot \text{bytes\_per\_float}$$

where the factor 2 accounts for storing both keys and values — this is the fundamental compute/memory trade-off underlying KV caching.

## Approach / Methodology

1.  **Data pipeline**: WikiText-2 (raw) is tokenized with the GPT-2 BPE tokenizer, concatenated with `<eos>` article boundaries, and packed into fixed-length, overlapping input/target windows for efficient batched training.
2.  **Baseline model**: a `TransformerLM` (Equinox module) is trained with AdamW, weight decay, linear warm-up, and cosine learning-rate decay, minimizing token-level cross-entropy.
3.  **KV-cache model**: a structurally identical `KVCacheTransformerLM` is built, with an attention/block/model implementation that dispatches between a full-sequence training path and a single-token cached inference path.
4.  **Correctness verification**: parameter counts and training-mode logits are checked for exact numerical agreement between the baseline and KV-cache model; greedy decoding with and without the cache is verified to produce identical token sequences on the same weights.
5.  **Benchmarking**: wall-clock generation time and tokens/second are compared between cached and non-cached decoding for a fixed number of generated tokens, using `jax.block_until_ready` for accurate timing under JAX's asynchronous dispatch.

## Key Terms

| Term | Meaning |
|------------------------------------|------------------------------------|
| Prefill | Parallel processing of the full input prompt to populate the cache |
| Decode | Sequential, one-token-at-a-time generation using cached states |
| Causal mask | Lower-triangular mask preventing attention to future tokens |
| KV cache | Stored key/value tensors reused across decoding steps |
| Weight tying | Reusing the token embedding matrix as the output projection |
| Perplexity | $e^{\text{NLL}}$, the standard language modeling evaluation metric |

## Results

- The KV-cache model is verified to be numerically equivalent to the baseline in training mode (max logit difference within floating-point tolerance).
- Greedy decoding with and without the cache produces identical generated token sequences on the same trained weights, confirming correctness.
- The cached decoding path achieves a measurable speedup in tokens/second over the non-cached path when generating a fixed number of new tokens, consistent with the theoretical reduction in redundant attention computation.
- Qualitatively, generated text reflects the capacity of a small (\~16M parameter), lightly trained model — locally coherent but prone to repetition under greedy decoding, which is expected behavior rather than an implementation defect.

## Libraries Used

- **JAX** — array computation and automatic differentiation.
- **Equinox** — PyTorch-like neural network modules for JAX.
- **Optax** — optimizers (AdamW) and learning-rate schedules.
- **Hugging Face `datasets`** — WikiText-2 loading.
- **Hugging Face `transformers`** (`GPT2TokenizerFast`) — tokenization.
- **NumPy** — data batching and array utilities.
- **Matplotlib** — training/validation curve visualization.
- **tqdm** — training progress bars.

## Repository Structure

```         
.
└── main.ipynb   # Full assignment notebook: data pipeline, baseline model,
                  # training, KV-cache implementation, correctness checks,
                  # and inference speed benchmark.
```

## References

- Google, *How to Scale Your Model — Inference*, JAX Scaling Book. <https://jax-ml.github.io/scaling-book/inference/>
- Raschka, S., *Coding the KV Cache in LLMs*. <https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms>
