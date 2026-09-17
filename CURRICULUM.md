# Deep Learning → LLMs: A Self-Study Curriculum

A from-scratch path rebuilding PyTorch reflexes and climbing to *the* Transformer
(Vaswani et al., 2017), with all modules eventually written by hand.

- **Pacing:** moderate — enough probing experiments to build intuition, not exhaustive.
- **Hardware:** Apple M4 Pro, MPS backend.
- **Rule for myself:** I write all harness, model, and dataset code. This doc is spec only.

---

## How to use this document

Each stage has:
- **Goal** — the one thing this stage is really teaching.
- **Toy problems** — with enough dataset detail to generate/load them myself.
- **Things to do** — an ordered checklist.
- **Done when** — concrete success criteria, so I know to move on.

Three habits to carry through *every* stage:
1. **Overfit a tiny batch first.** If a model can't memorize ~10 examples to ~100%, it has a
   bug — not a learning-rate problem. This is the single best debugging tool here.
2. **Reference-check from-scratch modules.** Copy weights from the `torch.nn` equivalent into
   my version and assert outputs match numerically (`torch.allclose`, atol ~1e-5). "Trains-ish"
   is not "correct."
3. **One new concept per step.** When building my own attention, do it on a problem already
   solved with a black box.

### MPS notes (M4 Pro)
- Set device once: prefer `mps` if `torch.backends.mps.is_available()`, else `cpu`.
- MPS is fp32-friendly; some ops fall back to CPU silently. Set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` while developing so you don't hit hard errors.
- For these toy sizes, CPU is often as fast as MPS. Don't fight it — correctness over speed
  until Stage 5.
- `gradcheck` needs float64, which MPS doesn't support → run gradient checks on **CPU**.

---

## Stage 0 — PyTorch reflexes

**Goal:** tensors, autograd, and the canonical training loop back in muscle memory.

### Toy problems
- **Regression:** fit `y = sin(x)` for `x ∈ [-2π, 2π]`. Sample ~1000 points, add small Gaussian
  noise (σ ≈ 0.1). 2-layer MLP, `tanh` or `ReLU`, MSE loss.
- **Classification:** two-moons (`sklearn.datasets.make_moons(n_samples=1000, noise=0.2)`) or
  spirals. Small MLP, cross-entropy. Plot the decision boundary.

### Things to do
- [ ] Write the canonical loop from memory: `zero_grad → forward → loss → backward → step`.
- [ ] Wrap data in a `Dataset` + `DataLoader`; do a proper train/val split.
- [ ] Make a clean device helper and move model + batches to it.
- [ ] Plot the fitted sine curve over the true curve; plot the moons decision boundary.
- [ ] Do **one** manual gradient check (CPU/float64) of a small custom function vs
      `torch.autograd.gradcheck`.

### Done when
- Sine fit is visually tight; moons val accuracy > 95%.
- I can write the training loop without looking anything up.

---

## Stage 1 — MLP + the fundamentals that transfer

**Goal:** meet LayerNorm, residuals, dropout, and init on an easy problem — they all return
in the Transformer.

### Dataset
- **MNIST** (`torchvision.datasets.MNIST`). 60k train / 10k test, 28×28 grayscale, 10 classes.
  Flatten to 784-dim vectors (no conv — keep it an MLP). Normalize with mean 0.1307, std 0.3081.

### Things to do
- [ ] Baseline MLP (e.g. 784→256→128→10) to a sane test accuracy (~97–98%).
- [ ] Add and observe the effect of, one at a time: dropout, BatchNorm, **LayerNorm**,
      a residual connection between two equal-width layers.
- [ ] Add a LR scheduler (cosine or step) and compare curves.
- [ ] Experiment: default init vs. a deliberately bad init (e.g. large σ) — watch it break.
- [ ] **Overfit drill:** take 100 examples, reach ~100% train accuracy. Internalize this.

### Done when
- Test accuracy ~98% with a tuned MLP.
- I can describe in one sentence each what LayerNorm and a residual connection *do* to the
  forward/backward signal.

---

## Stage 2 — Sequence models with high-level abstractions

**Goal:** sequence modeling using `nn.RNN` / `nn.LSTM` / `nn.GRU` as black boxes. Feel the
vanishing-gradient story directly.

### Toy problems (rough order of difficulty)
- **Running sum / parity:** input a length-`T` sequence of bits; output the running sum (regression)
  or parity (binary). Generate on the fly. Start `T=10`, push to `T=50`.
- **Copy / delay:** echo the input sequence after a delay of `k` steps. Vocab of ~8 symbols + a
  blank. Classic memory probe; try `k=5`, then `k=20`.
- **Adding problem** (Hochreiter & Schmidhuber): sequence of length `T` of pairs `(value, marker)`.
  `value ∈ [0,1]` uniform; exactly two positions have marker=1, rest 0. Target = sum of the two
  marked values. Try `T=50, 100, 200`. **Vanilla RNN struggles; LSTM handles it** — that's the lesson.
- **Char-level LM:** Tiny Shakespeare (~1 MB plain text, single file, widely mirrored as
  `tinyshakespeare`/`input.txt`). Char vocab (~65 symbols). Train next-char prediction; sample text.

### Things to do
- [ ] Implement variable-length batching: `pad_sequence` + `pack_padded_sequence`.
- [ ] Understand `batch_first`, and the shapes of `hidden`/`cell` state.
- [ ] On the **adding problem**, run vanilla RNN vs LSTM at `T=100,200`; plot loss. See RNN stall.
- [ ] Char-LM: implement teacher forcing for training; implement autoregressive sampling
      (temperature, greedy vs sampled) for generation.
- [ ] Sample a few hundred characters from the trained Shakespeare model — confirm it learns
      word-ish structure and punctuation.

### Done when
- LSTM solves the adding problem at `T=200` where vanilla RNN does not.
- Char-LM produces plausible-looking (not necessarily good) Shakespeare-ish text.

---

## Stage 3 — My own RNN Module

**Goal:** drop a level. Own the `nn.Parameter`s and write the recurrence loop in Python.

### Toy problems
Reuse Stage 2 problems — the point is matching the black box, not new data.

### Things to do
- [ ] Subclass `nn.Module`; implement a **vanilla RNN cell** holding my own weights/biases.
- [ ] Wrap it in a loop over time to process a full sequence (handle `batch_first`).
- [ ] Implement an **LSTM cell** (input/forget/cell/output gates) the same way.
- [ ] **Reference check:** copy weights from `torch.nn.RNNCell` / `nn.LSTMCell` into my modules
      and assert outputs match (`torch.allclose`, atol ~1e-5) on random inputs.
- [ ] Retrain my LSTM on the adding problem + char-LM; confirm parity with Stage 2 within noise.

### Done when
- Numerical match against the torch cells with copied weights.
- My hand-written LSTM reproduces the Stage 2 results.

---

## Stage 4 — Attention, before Transformers

**Goal:** the conceptual bridge. Build attention on top of an RNN seq2seq so "attention is all
you need" lands properly later.

### Toy problems (deterministic, infinite data, easy to eyeball)
- **Number → words:** `"213"` → `"two hundred thirteen"`. Generate by sampling integers in a range
  (e.g. 0–9999) and rendering with a small rule-based converter I write. Source vocab = digits;
  target vocab = number words.
- **Date normalization:** `"March 3, 2020"` → `"2020-03-03"`. Generate dates, render in several
  human formats (`"3/3/2020"`, `"March 3, 2020"`, `"3 Mar 2020"`), target = ISO format. Char-level.

### Things to do
- [ ] Build an RNN/LSTM **encoder–decoder** (seq2seq) with teacher forcing.
- [ ] Add **Bahdanau (additive) attention** in the decoder over encoder hidden states.
- [ ] Implement greedy decoding for inference.
- [ ] **Visualize the attention matrix** for a few examples — confirm it aligns sensibly
      (e.g. output digits attend to the right input span).
- [ ] (Optional) compare with/without attention on the longer date formats.

### Done when
- Near-perfect accuracy on the toy task (these are learnable to ~100%).
- Attention heatmaps look like meaningful alignments, not noise.

---

## Stage 5 — The Transformer, from scratch

**Goal:** build "Attention Is All You Need" (2017) myself — no `nn.Transformer`. Unit-test each
piece before assembling.

### Build order (test each in isolation)
1. [ ] **Scaled dot-product attention** — `softmax(QKᵀ/√d_k)V`, with optional mask.
2. [ ] **Multi-head attention** — the reshape/transpose into heads and back. The fiddly part.
3. [ ] **Positional encoding** — sinusoidal (fixed). Plot it as a heatmap to confirm.
4. [ ] **Position-wise FFN** + the **sublayer wrapper** (residual + LayerNorm). Decide and note
       pre-LN vs post-LN (2017 paper is post-LN).
5. [ ] **Encoder block**, then **decoder block** (causal self-attn mask + cross-attn).
6. [ ] **Full encoder–decoder**, with source/target embeddings and weight tying.

### Toy problems (in order)
- **Copy / reverse:** map a random integer sequence to itself / its reverse. Pure plumbing
  sanity check — should hit ~100%. If not, the masks or shapes are wrong.
- **Sort:** map a sequence of integers to its sorted version.
- **Toy translation:** reuse the Stage 4 date/number task for the full encoder–decoder with
  cross-attention.
- **Decoder-only char LM:** Tiny Shakespeare again, now with a GPT-style causal decoder stack.
  This is the natural pivot toward modern LLMs.

### Things to do
- [ ] **Reference checks:** my scaled-dot-product attention vs `F.scaled_dot_product_attention`;
      my encoder block vs `nn.TransformerEncoderLayer` with copied weights.
- [ ] Get **causal masking** right — verify position `t` cannot attend to `>t` (test by perturbing
      a future token and checking earlier outputs don't change).
- [ ] Implement learning-rate **warmup + inverse-sqrt decay** (the paper's schedule).
- [ ] Train copy/reverse to ~100%; then sort; then toy translation.
- [ ] Build the decoder-only char LM and sample text; compare quality to the Stage 2 LSTM.

### Done when
- Copy/reverse and sort hit ~100%.
- Toy translation matches or beats the Stage 4 attention seq2seq.
- I've built every sublayer myself and reference-checked the core ones.

---

## After Stage 5 — natural next steps (LLM-ward)

Not part of the core path, but where this leads:
- BPE / byte-level tokenization (replace char vocab).
- Pre-LN, RMSNorm, GELU, rotary position embeddings (RoPE) — the modern deltas vs 2017.
- KV-cache for fast autoregressive generation.
- Scaling the decoder-only LM on a larger corpus; sampling strategies (top-k, top-p).

---

## Personal log

Keep a running note per stage: what broke, the bug, and the fix. The bug list is the real
learning artifact.

- **Stage 0:**
- **Stage 1:**
- **Stage 2:**
- **Stage 3:**
- **Stage 4:**
- **Stage 5:**
