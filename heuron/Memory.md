# Heuron Research Group - Memory

## Environment
- Machine: Cineca Leonardo, 4xA100 64GB GPUs
- SLURM: partition=boost_usr_prod, account=IscrC_YENDRI, qos=boost_qos_dbg
- Max training time: 600s (10 min), max eval time: ~10 min, total session: 30 min
- Working file: `heuron/train_gpt_heron.py` (ONLY file we can modify)
- Launch command: `TRAIN_SCRIPT=heuron/train_gpt_heron.py salloc ... bash scripts/train_4xA100_interactive.sh`

## Model Architecture (Current Best - heron_2_4.py)
- 11 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP (1536 hidden)
- Vocab: 1024 (SentencePiece), tied embeddings
- Activation: LeakyReLU(0.5)^2
- Battle 1 gates: MLP output gate (12->1), attention gate (12->8), input-dependent SmearGate (12->1)
- Battle 2 ME: secondary embedding (1024x128) with per-layer gated injection (24->1 gate per layer)
- Features: SmearGate, BigramHash(1536), XSA last 4 layers, partial RoPE(16), LN scale, VE on layers 9/10, EMA(0.997), SWA(50), late QAT at 0.15
- ~27.1M params, int6+lzma compressed ~14.0MB

## Admin's Records
- Admin's best on 4xA100: 1.2061 BPB (final_int6_sliding_window, "prores" config)
- Admin pushed from 1.24 baseline to 1.2285 best with architecture changes only
- Current heron.py is initialized to "record before the last" with smaller warmdowns

## Research Battle 1: Gates
Focus: Where and how to use small gates to accelerate convergence.
Key insight from modded-nanogpt: 12-dim input-dependent sigmoid gates are extremely parameter-efficient (~72 params per gate) and outperform full-dim gates.

### Gate Types from modded-nanogpt
1. **Attention Gate**: `sigmoid(linear(x[:12]))` per-head, zero-init -> starts at 0.5
2. **VE Gate**: `2*sigmoid(linear(cat(x[:6], ve[:6])))` -> starts at 1.0
3. **SmearGate**: `lambda * sigmoid(linear(x[:12]))` with lambda init=0 -> starts off
4. **Skip Gate**: `sigmoid(lambda) * 2 * sigmoid(linear(x0[:12]))` -> starts ~0.18
5. **Scalar lambdas**: per-layer residual/output scaling (post_lambdas, resid_lambdas)

### Key Principles
- Zero/small init so gates start as no-op (model starts simple, adds complexity)
- Double/triple gating compounds control with minimal params
- 12 dims >> full dim for gate inputs (Muon optimization works better on near-square matrices)
- Input-dependent gates > fixed scales for attention modulation

## What Has Been Tried
### Exp 0: Baseline (heron.py unmodified)
- 1633 steps in 600s, 368ms/step, val_bpb=1.2382 (sliding window int6)

### Exp 1: Combined Lightweight Gates (heron4.py) - WIN: -0.0064 BPB
- Added to each Block:
  - MLP output gate: `mlp_gate_w = zeros(1, 12)` -> `2*sigmoid(F.linear(x_out[:12], w))` per-token scalar
  - Attention gate: `attn_gate_w = zeros(num_heads, 12)` -> `2*sigmoid(F.linear(x[:12], w))` per-head
- Both start at 1.0 (no-op) via 2*sigmoid(0)
- +1188 params total (negligible), 3% slower per-step
- val_bpb=1.2318 (sliding window int6), despite 50 fewer steps
- Added "mlp_gate" to CONTROL_TENSOR_NAME_PATTERNS

### Exp 2: + Input-dependent SmearGate (heron5.py) - BEST: -0.0083 BPB
- Added `smear_input_w = zeros(1, 12)` to SmearGate class
- Forward: `g = g_base * (2*sigmoid(F.linear(x[:12], smear_input_w)))` 
- Multiplicative modulation: base per-dim gate * per-token input-dependent scalar
- +12 params, val_bpb=1.2299 (sliding window int6)
- Must manually add to scalar_params in main()

### Exp 3: Skip Connection Gate (heron6.py) - FAILED: +0.0054 worse than Exp2
- Added input-dependent gate on U-Net skip connections
- Skip connections are already well-regulated by learned per-dim weights
- Extra gating introduces noise -> regression

### Exp 4: Per-dim MLP Gate (heron7.py) - FAILED: +0.0122 worse
- Changed MLP gate from (1, 12) to (512, 12) -> per-dim gating
- 67K extra params, 5% slower per-step
- 12-dim input cannot meaningfully gate 512 separate dimensions
- Scalar gate (12->1) is the sweet spot

### Exp 5: 16-dim Gate Inputs (heron8.py) - FAILED: +0.0044 worse than Exp2
- Widened all gate inputs from 12 to 16 dims
- Marginal difference, 12 dims is optimal
- More dims = more params to learn but gate decisions don't need more info

## Confirmed Principles
1. **12 input dims is the sweet spot** (confirmed by Exp 5 failure with 16)
2. **Scalar output gates beat per-dim** (confirmed by Exp 4 failure with 512)
3. **Gate existing flows, don't disrupt skip connections** (confirmed by Exp 3 failure)
4. **Total gate overhead must be < 1% of params** (~1200 new params on 27M model)
5. **2*sigmoid(0)=1.0 initialization is correct** for no-op start

## Research Battle 2: Multiple Embeddings
Focus: Whether additional embedding tables per-layer can improve BPB.

### ME Design from modded-nanogpt
1. **Secondary embedding**: nn.Embedding(vocab_size, me_dim), me_dim=128
2. **Padding**: F.pad(emb, (0, model_dim - me_dim)) to match residual stream width
3. **Per-layer scale**: zero-init scalar, model learns magnitude per layer
4. **Per-layer gate**: reads cat(x[:12], me[:12]) -> 24-dim input -> 1 scalar, 2*sigmoid(0)=1.0 init

### What Has Been Tried
- B2-1: ME dim=128, ungated -> 1.2292 (small win vs 1.2299 gates)
- B2-2: ME dim=64 -> 1.2332 (worse, too few dims)
- B2-3: ME hash vocab=512 -> 1.2330 (worse, loses token identity)
- B2-4: **Gated ME dim=128 -> 1.2279 (BEST, -0.0020 vs gates, beats admin 1.2285)**
- B2-5: Input Projections -> 1.2451 (terrible, no new info)
- B2-6: ME 2x128 ungated -> crashed (SLURM timeout)
- B2-7: Gated ME 2x128 -> 1.2342 (worse, 2 embeds add noise)
- B2-8: Gated ME dim=192 -> 1.2295 (close, 128 is optimal)

### Confirmed Principles
1. **Separate embedding > projections**: ME must provide genuinely independent features
2. **Gating ME injection is crucial**: +0.0013 BPB vs ungated
3. **128 dims is the sweet spot** for ME (64 too small, 192 marginal)
4. **One extra embedding is optimal** (two add noise + slow training)
5. **Full vocab required** (hashing loses token identity)
6. **Zero-init for both scale and gate** is correct initialization

## Untried Ideas for Next Researcher
- VE + ME synergy: expand VE layers alongside ME
- ME with different init (larger std, e.g., 0.05)
- Per-dim ME scales instead of scalar (128 params/layer)
- ME decay in later layers
- Residual stream amplification (resid_lambdas)
- ME with bigram context (hash(token, prev_token))
- Ablation: run gates-only vs ME-only to isolate contributions

## Research Battle 3: Attention Residuals & Similar
Focus: Real Attention Residual (DenseFormer), Block Attention Residual, Gated Attention Residual.

### Key Implementation Lesson (CRITICAL)
**DO NOT use raw `nn.Parameter(torch.zeros(...))` scalars optimized by Adam (scalar_lr=0.025) for unbounded contributions.**
Adam normalizes gradients, so scalars grow ~lr/step. After ~50 steps a zero-init scalar reaches ~1.25,
and with N layers each adding `~scalar * (layer output)`, the residual stream explodes ~Nx, destroying training.
Previous session lost all 10 attempts to this bug (BPB 1.27–1.30).

**Fix used in Battle 3**: parametrize as `dense_logits = nn.Parameter(torch.full((n,n), -6.0))` and use
`sigmoid(dense_logits)`. sigmoid(-6) ≈ 0.0025 (~zero start) and sigmoid is bounded in [0,1] so no explosion.

### What Has Been Tried (sigmoid-bounded)
- B3-1: Real DenseFormer on baseline -> 1.26437 (+0.0262 vs heron.py 1.2382). WORSE
- B3-2: Block Dense on baseline -> 1.29724 (+0.0590). WORSE (worst on baseline)
- B3-3: Gated Dense on baseline -> 1.25946 (+0.0213). WORSE
- B3-4: Real DenseFormer on record -> 1.25119 (+0.0233 vs heron_2_4 1.2279). WORSE
- B3-5: Block Dense on record -> 1.24209 (+0.0142). WORSE (best of 3 record variants)
- B3-6: Gated Dense on record -> 1.25154 (+0.0237). WORSE
- B3-7: Block Dense + record + init=-3.0 -> 1.24978 (+0.0219). WORSE than B3-5 (-6.0 init = 1.24209). Stronger initial dense hurts.

### Confirmed Principles (Battle 3)
1. **Sigmoid bound is necessary** for any added "scalar" parameter optimized by Adam.
   `dense_logits` init at -6 keeps weights near zero at start.
2. **Dense connections add 16-28% step-time overhead** (424-492ms vs ~380ms baseline).
   Stack + einsum implementation; explicit Python loop accumulation triggers InductorError.
3. **Block Dense behaves opposite on the two bases.**
   - On baseline (no ME): WORST (+0.059) — restricting to within-block starves the encoder→decoder
     path (only U-Net skip remains).
   - On record (has ME): BEST of 3 (+0.014) — ME provides the global per-layer signal so local
     dense within blocks complements rather than competes.
4. **None of the three techniques win on this regime.** The architecture is already heavily skip/ME
   connected; a 16MB / 600s training budget penalizes any extra compute that doesn't pay back per step.
5. **The 12-dim per-target gate** (vs per-pair) avoids the OOM kernel from prior session and compiles fine.

## Battle 3 Extension: U-Net Skip Ablation
Focus: Test whether dense connections were redundant with U-Net skips by removing skips first.

### What Has Been Tried
- B3-8: NO U-Net skips on baseline -> **1.2323** (-0.0059 vs baseline 1.2382). **WIN**.
- B3-9: NO skips + Real Dense on baseline -> 1.4261 (severely undertrained, 685 steps in 600s).
  Eager-mode dense add via `@torch.compiler.disable` ran at 877 ms/step (2.4× slower than B3-8).
  Per-step val_bpb at step 685 (1.3098) ~= B3-8 at step 1000 (1.3105) → ~46% per-step training-
  efficiency improvement from dense, but eager-mode overhead wipes it out in wallclock.
- B3-10: NO U-Net skips on record -> 1.2400 (+0.0121 vs record 1.2279). **REGRESSION**.

### Confirmed Principles (Battle 3 Extension)
6. **U-Net skip removal is base-dependent.** Helps the bare baseline (-0.0059), hurts the record (+0.0121).
   On the baseline, the per-channel `skip_weights` (init at zero) actively pull down the residual stream's
   effective rank. On the record, gated ME compensates and additionally cooperates with skips, so
   removing them strips signal.
7. **Dense connections actually work per-step in the no-skip regime.** First time in Battle 3 a dense
   variant is per-step better than its base. The fully-residual stream of the no-skip baseline gives
   dense connections something useful to do.
8. **Triton kernel fusion budget is a real constraint.** Without skips, inductor over-fuses dense
   bmm/einsum with rms_norm into a single triton kernel exceeding A100 shared-memory (Required:168k >
   Hardware:166k). The skip op `x += skip_weights[i] * skips.pop()` had been an implicit fusion barrier.
   Workarounds: scalar/per-channel barrier params didn't help; Python accumulator triggers
   `FusedMixOrderReductions` assertion. Working solution: `@torch.compiler.disable(recursive=False)`
   on a free function, requires `fullgraph=False` everywhere, costs 2.4× step time.
9. **B3-8 is the new best baseline-class architecture** (1.2323). Future Battle 4 experiments that
   previously used `heron.py` as their base should consider B3-8 instead. Does NOT replace the record
   1.2279 (which uses skips and gated ME together).
