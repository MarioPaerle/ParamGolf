# Heuron Research Group -- Research Recap

## Project Overview

This is the Heuron Research Group entry in the **OpenAI Parameter Golf** competition. The objective is to train the best language model that fits within a **16 MB** binary, evaluated by **bits-per-byte (BPB)** on the FineWeb validation set. All training runs execute on **Cineca Leonardo with 4x A100 GPUs**.

---

## Research Battle 1: Input-Dependent Gates

We investigated where and how to use small, input-dependent gates to accelerate convergence and improve final BPB.

### Experiments Run and Results

| # | Experiment | File | final\_int6\_sw BPB | vs Baseline | Status |
|---|-----------|------|---------------------|-------------|--------|
| 0 | Baseline (unmodified) | train\_gpt\_heron.py | 1.2382 | -- | Reference |
| 1 | MLP(12->1) + Attn(12->8) gates | train\_gpt\_heron4.py | 1.2318 | -0.0064 | WIN |
| 2 | Exp1 + Input-dependent SmearGate | train\_gpt\_heron5.py | 1.2299 | -0.0083 | **BEST** |
| 3 | Exp2 + Skip connection gate | train\_gpt\_heron6.py | 1.2353 | -0.0029 | WORSE (skip gate hurts) |
| 4 | Per-dim MLP gate (12->512) | train\_gpt\_heron7.py | 1.2421 | +0.0039 | WORSE (too many params, slower) |
| 5 | 16-dim gate inputs (vs 12) | train\_gpt\_heron8.py | 1.2343 | -0.0039 | WORSE (12 dims optimal) |

---

### What Worked: The Winning Recipe (heron5.py)

**1. Lightweight MLP output gate** -- 12 params per layer

In `Block.__init__`:
```python
mlp_gate_w = zeros(1, 12)
```
In forward:
```python
mlp_g = 2.0 * sigmoid(F.linear(x_out[:12], mlp_gate_w))
```
Gates MLP output per-token with a single scalar derived from the first 12 dims of the residual stream.

**2. Lightweight attention gate** -- 96 params per layer

In `CausalSelfAttention.__init__`:
```python
attn_gate_w = zeros(num_heads, 12)
```
In forward:
```python
attn_g = 2.0 * sigmoid(F.linear(x[:12], attn_gate_w)).unsqueeze(-1)
```
Gates attention output per-head, allowing each head to be dynamically scaled per-token.

**3. Input-dependent SmearGate** -- 12 params total

Added to `SmearGate`:
```python
smear_input_w = zeros(1, 12)
```
Multiplies the base fixed gate by `2.0 * sigmoid(F.linear(x[:12], smear_input_w))` to make smearing per-token adaptive.

---

### What Did NOT Work

1. **Skip connection gate (heron6.py):** Gating U-Net skip connections with an input-dependent gate HURT performance. The skip connections are already well-regulated by learned per-dim `skip_weights`.

2. **Per-dim MLP gate, 12->512 (heron7.py):** Too many degrees of freedom for the 12-dim input. Added 5% runtime overhead and produced worse BPB. The scalar gate (12->1) is the sweet spot.

---

### Key Principles Discovered

- **Small gates with zero-init are powerful.** `2 * sigmoid(0) = 1.0` makes gates start as identity (no-op), letting the model gradually learn to modulate.
- **12 input dimensions is the sweet spot.** Reading the first 12 dims of the residual stream provides enough information for gating. 16 dims (being tested in heron8) may help marginally.
- **Scalar gates > per-dim gates.** A single per-token scalar (12->1) works better than per-dim (12->512) for MLP gating. The information bottleneck of 12 dims limits how many output dims can be meaningfully gated.
- **Gate existing flows, don't disrupt them.** Block-level gates (MLP, attention) work because they modulate existing well-behaved outputs. Skip gates failed because they disrupted the U-Net information pathway.
- **Total new params must be tiny.** The winning config adds only ~1200 params to a 27M model. Gates that add >60K params (per-dim) degrade performance.

---

## Model Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP (1536 hidden)
- Activation: LeakyReLU(0.5)^2
- Features: SmearGate, BigramHash(1536), XSA (last 4 layers), partial RoPE(16), LN scale, VE on layers 9/10
- EMA(0.997), SWA(50), late QAT at 0.15
- Int6 quantization with LZMA compression
- Training: 600s wall clock on 4x A100, ~1580--1633 steps, ~380 ms/step

---

## Important Implementation Notes

- New gate params **MUST** be added to `CONTROL_TENSOR_NAME_PATTERNS` for proper quantization handling.
- New non-block params (like SmearGate additions) **MUST** be manually added to `scalar_params` in `main()`.
- The `replicated_params.extend(scalar_params)` call ensures distributed all-reduce works correctly.
- `torch.compile(fullgraph=True)` requires all ops to be traceable. Simple linear + sigmoid gates compile fine.
- Compilation adds 2--3 min overhead per run. Do not disable it for real runs.

---

## Files Overview

| File | Description |
|------|-------------|
| `train_gpt_heron.py` | Baseline (untouched reference) |
| `train_gpt_heron2.py` | MLP gate only (ablation) |
| `train_gpt_heron3.py` | Attention gate only (ablation) |
| `train_gpt_heron4.py` | MLP + attention gates combined (first WIN) |
| `train_gpt_heron5.py` | **BEST: MLP + attn + SmearGate enhancement** (1.2299 BPB) |
| `train_gpt_heron6.py` | Skip gate (FAILED) |
| `train_gpt_heron7.py` | Per-dim MLP gate (FAILED) |
| `train_gpt_heron8.py` | 16-dim gate inputs (RUNNING) |

---

## Suggested Next Experiments

1. **Check heron8 results** -- compare 16 vs 12 dim gate inputs.
2. **Value embedding gate** -- input-dependent gating of VE injection in layers 9/10.
3. **Residual stream amplification** -- per-layer learned scaling of residual (like modded-nanogpt `resid_lambdas` init at `sqrt(1.1)`).
4. **Gate attention pre-norm** -- gate the input to attention, not the output.
5. **"Backout" mechanism** -- subtract a gated fraction of middle-layer activations from the final output.
6. **Ablation runs** -- run heron2 (MLP only) and heron3 (attn only) to quantify individual gate contributions.

---

## Research Battle 2: Multiple Embeddings

We investigated whether injecting additional embedding representations per-layer could improve BPB. Starting from the Battle 1 winner (heron5.py, gates, 1.2299 BPB).

### Experiments Run and Results

| # | Experiment | File | val\_bpb @1000 | final\_int6\_sw BPB | vs Gates Baseline | Status |
|---|-----------|------|---------------|---------------------|-------------------|--------|
| B2-0 | Baseline (heron5 gates) | train\_gpt\_heron5.py | 1.3040 | 1.2299 | -- | Reference |
| B2-1 | ME dim=128 | train\_gpt\_heron\_2\_1.py | 1.3019 | 1.2292 | -0.0007 | WIN |
| B2-2 | ME dim=64 | train\_gpt\_heron\_2\_2.py | 1.3066 | 1.2332 | +0.0033 | WORSE |
| B2-3 | ME hash vocab=512 | train\_gpt\_heron\_2\_3.py | 1.3073 | 1.2330 | +0.0031 | WORSE |
| B2-4 | **Gated ME dim=128** | train\_gpt\_heron\_2\_4.py | **1.3006** | **1.2279** | **-0.0020** | **BEST** |
| B2-5 | Input Projections | train\_gpt\_heron\_2\_5.py | 1.3153 | 1.2451 | +0.0152 | TERRIBLE |
| B2-6 | ME 2x128 (no gate) | train\_gpt\_heron\_2\_6.py | 1.3025 | crashed | -- | CRASHED |
| B2-7 | Gated ME 2x128 | train\_gpt\_heron\_2\_7.py | 1.3024 | 1.2342 | +0.0043 | WORSE |
| B2-8 | Gated ME dim=192 | train\_gpt\_heron\_2\_8.py | 1.3008 | 1.2295 | -0.0004 | Close but worse |
| B2-9 | Gated Hash ME (512) | train\_gpt\_heron\_2\_9.py | 1.3067 | crashed | -- | SLURM timeout during sw eval |
| B2-10 | Gated Bigram ME (1536) | train\_gpt\_heron\_2\_10.py | 1.3046 | 1.2345 | +0.0046 | Bigram context helps vs plain hash |
| B2-11 | Ungated Bigram ME (1536) | train\_gpt\_heron\_2\_11.py | 1.3054 | 1.2381 | +0.0082 | ~= original baseline |

---

### What Worked: The Winning Recipe (heron\_2\_4.py)

**1. Secondary embedding table** -- 131,072 params

In `GPT.__init__`:
```python
self.me_embeds = nn.ModuleList([nn.Embedding(vocab_size, me_dim)])  # me_dim=128
nn.init.normal_(emb.weight, std=0.01)
```
One extra embedding table of dimension 128 provides a different "view" of each token.

**2. Per-layer zero-init scales** -- 11 params

```python
self.me_layer_scales = nn.ParameterList([
    nn.Parameter(torch.zeros(1, dtype=torch.float32))
    for _ in range(num_layers)
])
```
Each layer learns its own magnitude for the ME injection. Zero-init means the model starts without ME and gradually learns to use it.

**3. Per-layer input-dependent gate** -- 264 params

```python
self.me_gate_ws = nn.ParameterList([
    nn.Parameter(torch.zeros(1, 24, dtype=torch.float32))
    for _ in range(num_layers)
])
```
Gate reads `cat(x[:12], me[:12])` (first 12 dims of both current activation and ME signal) to produce a per-token scalar. Init at `2*sigmoid(0)=1.0` (no-op).

**4. Injection at each layer input**

```python
def _inject_me(self, x, layer_idx, me_signals):
    gate_input = torch.cat([x[..., :12], sig[..., :12]], dim=-1)
    g = 2.0 * torch.sigmoid(F.linear(gate_input, gate_w))
    x = x + scale * g * sig
    return x
```
Applied before each block in both encoder and decoder.

**5. Padding for compile-safety**

```python
xe_padded = F.pad(emb(input_ids), (0, model_dim - me_dim))  # (B, T, 512) with zeros after 128
```
Zero-pads the 128-dim embedding to full 512-dim for addition to the residual stream. Only the first 128 dims carry the ME signal.

---

### What Did NOT Work

1. **ME dim=64 (B2-2):** Smaller dimension loses too much capacity. The embedding needs at least 128 dims to capture useful alternative token representations.

2. **ME hash vocab=512, ungated (B2-3):** Hashing 1024 tokens to 512 buckets causes collisions -- ~2 tokens share each entry. The secondary embedding can't distinguish them, injecting misleading signal. Result: 1.2330, **worse than no ME at all** (1.2299).

3. **Gated Hash ME (B2-9):** Adding an input-dependent gate to the hash approach (gate reads x[:12] which has full token identity) partially recovers -- the gate can suppress ME when collision makes it ambiguous. At step 1000: 1.3067, better than ungated hash (1.3073) but still far worse than full-vocab gated (1.3006). Crashed during sliding window eval (SLURM timeout).

4. **Bigram ME (B2-10, B2-11):** Using bigram hash `xor(36313*token[t], 27191*token[t-1]) % 1535` as ME index gives context-dependent secondary embeddings (1536 entries). Gated bigram (B2-10): 1.2345; ungated bigram (B2-11): 1.2381. Better than plain hash (context helps) but still significantly worse than full-vocab ME. The bigram table has 50% more entries (1536 vs 1024) but many entries are rarely seen, so they don't train well in 600s. The gate helps (+0.0036 for bigram).

5. **Input Projections (B2-5):** Projecting x0 via a learned matrix (`xe = x0 @ W_proj`) instead of a separate embedding table is TERRIBLE (+0.015 BPB). A projection of x0 doesn't add genuinely new information -- it's a linear transformation of the same data. A separate embedding table provides independent learned features.

6. **ME 2x128, both ungated (B2-6) and gated (B2-7):** Two extra embeddings hurt. The second embedding adds noise rather than signal. One high-quality secondary embedding is optimal.

7. **ME dim=192 (B2-8):** Very close to dim=128 at step 1000 (1.3008 vs 1.3006) but slightly worse in the final result (1.2295 vs 1.2279). The extra 64 dims add 65K params without proportional benefit, and the model has slightly fewer tokens to compress.

---

### Key Principles Discovered

- **Separate embedding table > projections.** The secondary embedding must provide genuinely independent learned features, not a transformation of the primary embedding.
- **Input-dependent gating is crucial for ME.** The gate allows per-token modulation of ME injection, adding +0.0013 BPB vs ungated ME. Gate reads both x and ME signal (24 input dims -> 1 scalar).
- **128 dims is the sweet spot for ME.** 64 dims is too small, 192 is marginally worse (fewer training steps), 128 hits the right balance.
- **One extra embedding is optimal.** Two embeddings add noise and slow training (404ms vs 392ms/step), resulting in worse BPB.
- **Full vocab is required.** Hashing to smaller vocab (B2-3, B2-9) loses per-token identity. Even with a gate that can see full token identity via x[:12], the gate can only suppress bad signals, not recover lost information. Half the vocab = significantly worse.
- **Bigram context is not enough.** Bigram ME (B2-10, B2-11) gives context-dependent embeddings via hash(token, prev\_token), but the larger table (1536) trains slowly and many entries are undersampled in 600s. Full-vocab unigram ME trains more efficiently.
- **Gating always helps.** Across all ME variants, adding the input-dependent gate improved results: ungated ME -> gated ME (-0.0013), ungated bigram -> gated bigram (-0.0036), ungated hash -> gated hash (improved at step 1000). The gate's ability to read x[:12] lets it make per-token injection decisions.
- **F.pad is compile-safe** for extending smaller embeddings to full model_dim.
- **Zero-init for both scale and gate** means ME starts as no-op, gradually learning to contribute.

---

### Cumulative Best: Gates + Gated ME

The winning configuration combines Battle 1 gates with Battle 2 gated ME:
- **MLP output gate** (12->1, per-token scalar): 12 params/layer
- **Attention gate** (12->8, per-head): 96 params/layer
- **Input-dependent SmearGate** (12->1): 12 params total
- **Secondary embedding** (1024 x 128): 131,072 params
- **Per-layer ME scales** (zero-init): 11 params
- **Per-layer ME gates** (24->1): 264 params

Total new params: ~132,500 on a 27M model (0.5%).

---

## Reference Points

| Source | BPB |
|--------|-----|
| Admin's best (4x A100, progressive batch resizing) | 1.2061 |
| Admin's best (4x A100, architecture-only changes) | 1.2285 |
| Leaderboard SOTA (TTT, Parallel Muon, 8x H100) | 1.1194 |
| Our best after Battle 1 (heron5.py, gates) | 1.2299 |
| **Our best after Battle 2 (heron\_2\_4.py, gated ME)** | **1.2279** |

---

## Model Architecture (Current Best)

- 11 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP (1536 hidden)
- Activation: LeakyReLU(0.5)^2
- Features: SmearGate (input-dependent), BigramHash(1536), XSA (last 4 layers), partial RoPE(16), LN scale, VE on layers 9/10
- **Battle 1 gates**: MLP output gate, attention gate, smear input gate
- **Battle 2 ME**: secondary embedding (128-dim) with per-layer gated injection
- EMA(0.997), SWA(50), late QAT at 0.15
- Int6 quantization with LZMA compression
- Training: 600s wall clock on 4x A100, ~1535 steps, ~392 ms/step
- ~27.1M params, ~14.0MB compressed

---

## Important Implementation Notes

- ME embedding weights go to `tok_params` (Adam optimizer with `tied_embed_lr`).
- ME layer scales and gate weights go to `scalar_params` (Adam optimizer).
- `me_layer_scale` and `me_gate_w` must be in `CONTROL_TENSOR_NAME_PATTERNS` for proper quantization handling.
- All ME params must be added to `replicated_params` (via scalar_params extend) for distributed all-reduce.
- The `F.pad` approach to extend me_dim to model_dim is compile-safe with `torch.compile(fullgraph=True)`.

---

## Files Overview

| File | Description |
|------|-------------|
| `train_gpt_heron.py` | Baseline (untouched reference) |
| `train_gpt_heron5.py` | Battle 1 BEST: gates (MLP + attn + SmearGate) |
| `train_gpt_heron_2_1.py` | ME dim=128, ungated (+0.0007 vs gates) |
| `train_gpt_heron_2_2.py` | ME dim=64 (WORSE) |
| `train_gpt_heron_2_3.py` | ME hash vocab=512 (WORSE) |
| `train_gpt_heron_2_4.py` | **BEST: Gated ME dim=128 (1.2279 BPB)** |
| `train_gpt_heron_2_5.py` | Input Projections (TERRIBLE) |
| `train_gpt_heron_2_6.py` | ME 2x128, ungated (CRASHED) |
| `train_gpt_heron_2_7.py` | Gated ME 2x128 (WORSE) |
| `train_gpt_heron_2_8.py` | Gated ME dim=192 (close, slightly worse) |
| `train_gpt_heron_2_9.py` | Gated Hash ME, vocab=512 (CRASHED during eval) |
| `train_gpt_heron_2_10.py` | Gated Bigram ME, vocab=1536 (better than hash, worse than full) |
| `train_gpt_heron_2_11.py` | Ungated Bigram ME, vocab=1536 (~= baseline) |

---

## Suggested Next Experiments

1. **Value Embedding + ME synergy** -- ME provides info to residual, VE provides info to attention values. They may complement each other. Try increasing VE layers or dim alongside ME.
2. **ME with different init** -- Try larger std for ME embedding init (e.g., 0.05 vs 0.01).
3. **Per-dim ME scales instead of scalar** -- The current per-layer scale is a single scalar. A per-dim scale (128 params/layer) could let different ME dimensions contribute differently at each layer.
4. **ME decay** -- Scale ME contribution down in later layers (later layers need less token identity signal).
5. **Residual stream amplification** -- Combine ME with per-layer learned residual scaling (modded-nanogpt resid_lambdas).
6. **Selective-layer ME** -- Only inject ME at certain layers (e.g., first half or last half) instead of all 11. This saves compute and might focus the signal.
7. **ME with learned per-dim gate** -- Instead of scalar gate (24->1), try a small per-dim gate (24->128) that can selectively amplify/suppress individual ME dimensions per-token.

---

## How to Run

```bash
TRAIN_SCRIPT=heuron/train_gpt_heron_2_4.py RUN_ID=my_exp \
  salloc \
    --partition=boost_usr_prod \
    --account=IscrC_YENDRI \
    --qos=boost_qos_dbg \
    --gres=gpu:4 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --mem=120G \
    --time=00:30:00 \
    bash scripts/train_4xA100_interactive.sh
```

Replace `train_gpt_heron_2_4.py` with the desired experiment file and `my_exp` with a descriptive run ID.

For parallel jobs (up to 4 at a time), use sbatch:

```bash
TRAIN_SCRIPT=heuron/train_gpt_heron_2_4.py RUN_ID=my_exp sbatch scripts/train_4xA100.sh
```

---

## Research Battle 3: Attention Residuals & Similar

We investigated whether adding DenseFormer-style residual connections between layers (Real Dense, Block Dense, Gated Dense) could improve BPB on either the baseline (`heron.py`, 1.2382) or the record (`heron_2_4.py`, 1.2279).

### Experiments Run and Results

| # | Experiment | File | Base | val_bpb @1000 | final_int6_sw BPB | vs Base | Status |
|---|-----------|------|------|---------------|-------------------|---------|--------|
| B3-0a | Baseline (no dense) | train_gpt_heron.py | -- | 1.3118 | 1.2382 | reference | Reference |
| B3-0b | Record (no dense) | train_gpt_heron_2_4.py | -- | 1.3006 | 1.2279 | reference | Reference |
| B3-1 | Real DenseFormer | train_gpt_heron_3_1.py | baseline | 1.3136 | 1.26437 | +0.0262 | WORSE |
| B3-2 | Block Attn Residual | train_gpt_heron_3_2.py | baseline | 1.4005 | 1.29724 | +0.0590 | WORSE (worst) |
| B3-3 | Gated Attn Residual | train_gpt_heron_3_3.py | baseline | 1.3103 | 1.25946 | +0.0213 | WORSE |
| B3-4 | Real DenseFormer | train_gpt_heron_3_4.py | record | 1.3007 | 1.25119 | +0.0233 | WORSE |
| B3-5 | Block Attn Residual | train_gpt_heron_3_5.py | record | 1.3013 | **1.24209** | +0.0142 | WORSE (best of 3) |
| B3-6 | Gated Attn Residual | train_gpt_heron_3_6.py | record | 1.3006 | 1.25154 | +0.0237 | WORSE |
| B3-7 | Block Attn Res init=-3 | train_gpt_heron_3_7.py | record | 1.3031 | 1.24978 | +0.0219 | WORSE than B3-5 |

### Critical Implementation Lesson

**DO NOT use raw `nn.Parameter(torch.zeros(...))` scalars optimized by Adam (scalar_lr=0.025) for unbounded contributions to the residual stream.**

Adam normalizes per-parameter gradients, so each step pushes a scalar by ~lr regardless of true gradient magnitude. After ~50 steps a zero-init scalar reaches ~1.25, and with N=11 layers each adding ~scalar × layer_output, the residual stream blows up ~10×. The previous session of this Battle lost all 10 runs to this bug (final BPB 1.27–1.30 across the board).

**Fix used in Battle 3 (final session)**: parametrize as `dense_logits = nn.Parameter(torch.full((n,n), -6.0))` and read `sigmoid(dense_logits)`. sigmoid(-6) ≈ 0.0025 (~zero start), bounded in [0, 1]. Even when Adam pushes the logit aggressively, contributions stay bounded.

**Compile lesson**: use `torch.stack(layer_outputs, dim=0)` + `torch.einsum('i,ibtd->btd', w, stacked)` for the dense accumulation. Iterative `x = x + da[j] * layer_outputs[j]` triggers Inductor "FusedMixOrderReductions" errors. Per-pair input-dependent gates `gate_w[i, :i]` shape (i, 12) trigger Triton OOM during autotune; use single per-target-layer gate `gate_w[i:i+1]` shape (1, 12) instead.

### Pseudocode of the (Failed) Best Attempt: Block Dense + Record (heron_3_5.py)

```python
# In GPT.__init__ (after the existing Battle 2 ME setup):
self.num_encoder_layers = num_layers // 2          # = 5
self.num_decoder_layers = num_layers - n//2        # = 6
self.block_dense_enc_logits = nn.Parameter(
    torch.full((num_encoder_layers, num_encoder_layers), -6.0, dtype=torch.float32)
)
self.block_dense_dec_logits = nn.Parameter(
    torch.full((num_decoder_layers, num_decoder_layers), -6.0, dtype=torch.float32)
)
# ALSO: append both to scalar_params and to CONTROL_TENSOR_NAME_PATTERNS

# In GPT.forward / forward_logits, replacing the encoder/decoder loops:
me_signals = self._compute_me_signals(input_ids, x.size(-1))
enc_outputs: list[Tensor] = []
dec_outputs: list[Tensor] = []
enc_w = torch.sigmoid(self.block_dense_enc_logits).to(dtype=x.dtype)
dec_w = torch.sigmoid(self.block_dense_dec_logits).to(dtype=x.dtype)
for i in range(self.num_encoder_layers):
    if i > 0:
        stacked = torch.stack(enc_outputs, dim=0)  # (i, B, T, D)
        w = enc_w[i, :i]                            # (i,)
        x = x + torch.einsum('i,ibtd->btd', w, stacked)
    x = self._inject_me(x, i, me_signals)
    x, raw_v = self.blocks[i](x, x0, ...)
    enc_outputs.append(x)
    skips.append(x)
for i in range(self.num_decoder_layers):
    bi = self.num_encoder_layers + i
    if skips:
        x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
    if i > 0:
        stacked = torch.stack(dec_outputs, dim=0)
        w = dec_w[i, :i]
        x = x + torch.einsum('i,ibtd->btd', w, stacked)
    x = self._inject_me(x, bi, me_signals)
    x, _ = self.blocks[bi](x, x0, ...)
    dec_outputs.append(x)
```

### Key Findings

1. **None of the three techniques win.** Even in the most charitable case (Block Dense on record), final BPB is +0.0142 worse than record. Step 1000 val_bpb is virtually identical to record (1.3013 vs 1.3006), but the gap opens during late training and the 16% slowdown means fewer steps in the budget.

2. **The damage is smaller on the record than on the baseline.** Because record has ME providing per-layer per-token information, the dense connections compete with that signal less destructively.

3. **Block Dense behaves OPPOSITE on the two bases**:
   - **Baseline (no ME)**: WORST (+0.0590). Restricting dense to within-block local connectivity starves the encoder→decoder cross-block path (only U-Net skip remains for that direction).
   - **Record (has ME)**: BEST of 3 (+0.0142). With ME providing global per-layer information, restricting dense to local connectivity stops it from competing with ME's signal.

4. **Stronger initial dense (init=-3.0) is WORSE than init=-6.0** (B3-7 vs B3-5: 1.24978 vs 1.24209). The "weakest possible" init is the best of the bad options. There is no init that lets dense connections improve this architecture.

5. **Step time slowdown alone explains 30-50% of the regression.** All variants run at 424–492 ms/step vs ~380 ms baseline → 16-28% fewer training steps in 600s. The dense connections do not pay back this cost per step.

6. **The architecture is already heavily skip- and ME-connected.** Adding more residual paths is redundant in this regime: U-Net skip (encoder↔decoder), `resid_mix` (per-layer learned residual scaling), and (on record) per-layer ME injection already provide rich information flow.

### Reference Points

| Source | BPB |
|--------|-----|
| Admin's best (4x A100, progressive batch resizing) | 1.2061 |
| Admin's best (4x A100, architecture-only changes) | 1.2285 |
| Leaderboard SOTA (TTT, Parallel Muon, 8x H100) | 1.1194 |
| Our best after Battle 1 (heron5.py, gates) | 1.2299 |
| **Our best after Battle 2 (heron_2_4.py, gated ME)** | **1.2279** |
| Best Battle 3 attempt (heron_3_5, Block Dense + record) | 1.24209 |
| Best Battle 3 baseline attempt (heron_3_3, Gated Dense) | 1.25946 |
| **Battle 3 ext. (heron_3_8, NO U-Net skips on baseline)** | **1.2323** |
| Battle 3 ext. (heron_3_10, NO skips on record) | 1.2400 |

**Battle 3 architecture-only result: no improvement. Current overall best remains heron_2_4.py at 1.2279 BPB.**

**Battle 3 *extension* result: removing U-Net skips on the unmodified baseline gives -0.0059 BPB
(1.2382→1.2323). This is the largest single-knob win against the baseline since Battle 1's gates.
The same change on the record (which has gated ME) is a regression (+0.0121).**

---

## Files Overview (Battle 3)

| File | Description |
|------|-------------|
| `train_gpt_heron_3_1.py` | Real DenseFormer on baseline (sigmoid-bounded) -> 1.26437 |
| `train_gpt_heron_3_2.py` | Block Dense on baseline -> 1.29724 (worst) |
| `train_gpt_heron_3_3.py` | Gated Dense on baseline -> 1.25946 |
| `train_gpt_heron_3_4.py` | Real DenseFormer on record -> 1.25119 |
| `train_gpt_heron_3_5.py` | Block Dense on record -> **1.24209** (best of 7) |
| `train_gpt_heron_3_6.py` | Gated Dense on record -> 1.25154 |
| `train_gpt_heron_3_7.py` | Block Dense on record, init=-3.0 (ablation) -> 1.24978 |
| `train_gpt_heron_3_8.py` | **NO U-Net skips control (baseline minus skips) -> 1.2323 (-0.0059)** |
| `train_gpt_heron_3_9.py` | NO skips + Real Dense, eager-mode dense add -> 1.4261 (severely undertrained, 685 steps; per-step better than B3-8) |
| `train_gpt_heron_3_10.py`| NO skips on record -> 1.2400 (+0.0121, regression) |

---

## Suggested Next Experiments (for future battles)

Battle 3 explored the local-residual-pattern direction and found it does not pay off here. Promising directions for future battles, in rough order of promise:

1. **Battle 4: ProRes / residual scaling.** Per-layer learned residual scales (modded-nanogpt `resid_lambdas`, init at sqrt(1.1)) — single scalar per layer, almost zero param cost, demonstrated to help in modded-nanogpt. Would not interfere with existing skip / ME structure.

2. **Battle 4: Late-layer specialization.** Currently the same Block class is used for all 11 layers; specializing the last 1-2 layers (e.g., wider MLP, different attention pattern) could pay off given that the residual stream has converged by then.

3. **Battle 5: TTT / continual updates during eval** — leaderboard SOTA uses Test-Time Training. Big jump (0.06 BPB possible), but compile-fragile.

4. **Battle 5: Parallel Muon for embeddings.** Modded-nanogpt uses Muon for embeddings via parallel scheme — could give ~0.005 BPB.

Notes:
- DO NOT revisit attention residuals / dense connections without a fundamentally different parametrization (e.g., learnable mask of dense positions, or dense via low-rank projection rather than identity-add). The 7 experiments here cover the standard space well.
- DO retain the sigmoid-bounded scalar parameter trick — use it any time you add a learnable contribution to the residual stream optimized by Adam.

### Important Battle 3 Extension Insights (2026-04-07)

5. **U-Net skip removal is base-dependent.** B3-8 (baseline minus skips) = 1.2323, **-0.0059** vs baseline 1.2382.
   B3-10 (record minus skips) = 1.2400, **+0.0121** vs record 1.2279. The same change helps the
   bare baseline and hurts the gated-ME record. Hypothesis: the original per-channel `skip_weights`
   (init at zero) actively pull down the residual stream's effective rank in the absence of stronger
   structural signal; gated ME compensates for this and additionally cooperates with skips, so removing
   skips strips signal in the record case.

6. **Battle 4 starting point candidate**: heron_3_8 (no-skip baseline at 1.2323) is the new
   best baseline-class architecture. Future Battle 4 experiments that previously used `heron.py`
   as their base should consider B3-8 instead. (Note: this does NOT replace the record — the record
   1.2279 still wins on absolute BPB and uses skips.)

7. **Dense connections actually work per-step when not redundant with skips.** B3-9 with eager-mode
   dense (compile-disabled free function) reached val_bpb 1.3098 at step 685, equivalent to B3-8's
   step ~1000. The 2.4× per-step compute overhead from eager mode wipes out the win in the
   wallclock-bounded final BPB (1.4261). A compile-friendly fast dense implementation (custom kernel,
   manual fusion barrier, or dense via low-rank projection) is the open lever for converting this
   per-step win into a wallclock-bounded win. Reserved for Battle 4.

8. **Triton kernel fusion budget is a real constraint.** Removing U-Net skips lets `torch.compile`
   inductor over-fuse the dense bmm/einsum with rms_norm into a single triton kernel that exceeds
   the hardware shared-memory limit (Required:168k > Hardware:166k on A100). The skip operation
   `x = x + skip_weights[i] * skips.pop()` had been acting as an implicit fusion barrier.
   Workarounds attempted: scalar barrier param, per-channel barrier param, Python accumulator loop
   (hits `FusedMixOrderReductions` assertion). Working solution: `@torch.compiler.disable(recursive=False)`
   on a free function (requires `fullgraph=False` everywhere, costs 2.4× step time).
