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

## Research Battle 4: Is what worked truly transferable? (NEW heuron baseline)
Focus: Take the new (much-improved) `train_gpt_heuron.py` baseline, fix FA3→SDPA, then test whether
the past Battle 1 (gates) and Battle 2 (gated ME) winners transfer onto it. The new baseline is
fundamentally different: ParallelMuon, MTP heads (off), GPTQ Hessian-aware int6 quantization with
autoregressive self-gen calibration, selective ±1 pruning, **XSA on ALL 11 layers** (vs old XSA-on-last-4),
bigram2048, VE on layers 9/10, MUON warmup 0.92→0.99 over 1500 steps, EMA(0.997), SWA from 1500,
late_qat 0.15. Old battles' winners are NOT present in this baseline.

### Required environment fix (not a research result)
The new baseline imports `flash_attn_interface` (FA3). FA3 is not installable on Cineca A100s (FA3 is
H100-only in practice). Replaced both attention call sites (`CausalSelfAttention.forward` and
`_HessianAttn.forward`) with `F.scaled_dot_product_attention(...)`, which dispatches to FlashAttention-2
on A100 because the script already calls `enable_flash_sdp(True)`. Maintained the (B,T,H,D) tensor
layout that the XSA path expects via `transpose(1,2).contiguous()`. Applied in-place to
`heuron/train_gpt_heuron.py`.

### What Has Been Tried (Battle 4)
- B4-0: Bare patched new baseline → **1.23260** (-0.0056 vs old heron baseline 1.2382, +0.0047 vs old
  record heron_2_4 1.2279). 1568 steps, 382 ms/step. THIS IS THE BATTLE 4 WINNER.
- B4-1: B4-0 + Battle 1 lightweight gates (attn_gate_w + mlp_gate_w + smear_input_w, all zero-init via
  2*sigmoid) → 1.23381 (+0.00121 vs B4-0). 1538 steps, 390 ms/step. **NOT TRANSFERABLE**.
- B4-2: B4-0 + Battle 2 gated ME (1024×128 secondary embedding, per-layer scale + (1,24) gate) →
  1.23338 (+0.00078 vs B4-0). 1508 steps, 398 ms/step. **NOT TRANSFERABLE**.
- B4-3: B4-0 + gates + gated ME (combined) → 1.23523 (+0.00263 vs B4-0). 1494 steps, 402 ms/step.
  Best LIVE pre-EMA val_bpb of all four (1.2331), but loses the wallclock budget battle. **NOT TRANSFERABLE**.

### Confirmed Principles (Battle 4)
1. **Past winners do NOT transfer cleanly.** All three modifications score WORSE than the bare new
   baseline on `final_int6_sliding_window`. The new baseline's design choices already absorb most of
   the value the old gates/ME captured (XSA-on-all-layers replaces gated ME's per-layer per-token
   information; MUON warmup replaces the gated initialization smoothing).
2. **Live (pre-EMA) BPB favors the additions, especially the combination.** B4-3 has the best live
   val_bpb at 1.2331. So gates DO still improve per-step quality on this baseline — they just don't
   improve it enough to overcome the step-count deficit they introduce.
3. **The slowdowns matter.** Each addition costs ~2-5% step time: B4-1 +1.8%, B4-2 +4.0%, B4-3 +5.0%.
   On a 600s wallclock cap this is 30-74 lost steps. With a marginal per-step improvement, total
   wallclock-bounded BPB regresses.
4. **GPTQ int6 roundtrip damage is similar across all variants** (~+0.0034 to +0.0051 added BPB from
   post_ema to int6_roundtrip). The added gate/ME tensors don't make the new GPTQ pipeline noticeably
   worse, despite small `(1,12)` and `(num_heads,12)` shapes. The new int4-band classification handles
   them via the int8/per-row path same as `me_embeds.weight` does in the old record.
5. **Sliding-window vs roundtrip improvement is consistent at ~−0.024 across runs**, confirming the
   sliding-window context-extension benefit is architecture-independent.
6. **Strong baselines absorb past wins.** This is the meta-lesson of Battle 4: when you upgrade your
   architecture significantly (XSA layer-coverage, optimizer warmup, smarter quantization, more
   embedding signal sources), simple-add patches that beat a weaker baseline often stop helping.
   Future research should focus on novel mechanisms that the new baseline does NOT already have, not
   on porting old wins.

### Untried for next agent
- Per-channel ME scales instead of scalar (might recover the gated-ME advantage if the issue is
  underexpression of the ME signal under the new training schedule)
- Larger ME std init (e.g., 0.05) to give the ME more starting signal so it doesn't rely on the slow
  zero-init scale ramp-up
- Stronger gate initialization for gates (e.g., init `attn_gate_w` to small positive values, so gate
  starts at 1.05-1.10 instead of 1.0, breaking the no-op symmetry from step 1)
- The B3-8 "no skip" finding adapted to the new heuron baseline (not tested in Battle 4 — separate
  follow-up)
- New mechanisms specific to the new baseline: maybe additional MUON warmup variants, different SWA
  schedules, alternative GPTQ calibration sequence lengths

## Research Battle 4 — Continuation: XSI experiments (Session 5)

User's new TODO: try XSI (Exclusive Input Attention) variants — orthogonalize layer output against
layer input (instead of against the value projection like XSA does).

### XSI variants implemented
- **5_1 (XSI v3, final-logit, parameterless)**: orthogonalize the final residual stream against `x0`
  before final_norm + lm_head. Just one ortho op per forward.
- **5_2 (XSI v2, layerwise k=4, parameterless, stop-grad on src)**: every 4 layers, orthogonalize
  current residual against the snapshot from k layers earlier. Stop-grad needed to keep the fused
  backward kernel under A100 SMEM ceiling (~167KB).
- **5_3 (XSI v1, naive block, helper-disabled)**: per-block, orthogonalize attn_out against block
  input, mlp_out against post-attention residual. Strongest dose. Requires `@torch._dynamo.disable`
  helper because even `.detach()` couldn't shrink the fused kernel enough.
- **5_7 (XSI v3 + learnable α, tanh-bounded, zero-init)**: same as 5_1 but the orthogonalization is
  scaled by `tanh(xsi_alpha)` with `xsi_alpha` a single scalar parameter init at 0. Recovers baseline
  at init, lets the model learn whether to opt in.

### Compile-time lesson (Scratch.md material too)
torch.compile fuses small element-wise ops (F.normalize + dot + sub + scale) into the surrounding
block's backward kernel. With the residual-stream connection, this kernel grows past A100 SMEM
(168KB / SM). Two recovery patterns:
- **`.detach()` on the projection target** (BYOL/SimSiam stop-grad): drops the cross-coupling
  gradient term, saves a few KB of register/SMEM. Often enough; not always (5_3 went from 170KB
  → 168KB, still 1KB over).
- **`@torch._dynamo.disable` on a small helper function**: forces eager execution for the few ops
  inside the helper, which breaks the fusion completely. Slightly slower but always works.
Prefer the helper approach when detach alone is insufficient or when you don't want to drop the
cross-coupling gradient.

### Results
- **5_1 (XSI v3, parameterless)**: final_int6_sw **1.24137** → **+0.00877 vs B4-0**. Same speed.
  Pure per-step quality regression — rigid orthogonalization against the input embedding hurts.
- **5_2 (XSI v2 layerwise k=4, parameterless, stop-grad)**: final_int6_sw **1.24715** →
  **+0.01455 vs B4-0**. Step_avg 403ms (+5% slower) → only 1489 steps vs 1568. Worse on every axis.
  Initial triton-OOM in backward fixed by `.detach()` on the saved residual snapshots.
- **5_3 (XSI v1 naive block, attn-only)**: BLOCKED — triton OOM in backward (different fused kernel
  than 5_2; even with `x_in.detach()` in the projection ref, the attn-out backward fuses with the
  projection sum and exceeds the 166912-byte shared-mem limit). Abandoned.
- **5_7 (XSI v3 + learnable α, single tanh-bounded scalar, init 0)**: final_int6_sw **1.23188** →
  **−0.00072 vs B4-0** ✅ **WIN.** Same speed (383ms/step). Cost: +1 scalar parameter `xsi_alpha`.
  Code: identical to 5_1 except `x = x − tanh(self.xsi_alpha) * <x, x0_n> * x0_n` before final_norm.
  **Reproduced** on a confirmation run: dup-5_7 = **1.23161** (within ±0.0003 noise).
- **5_8 (XSI v3 + per-channel α, D=512 params)**: final_int6_sw **1.23423** → **+0.00163 vs B4-0**,
  **worse than 5_7 by +0.00262**. Per-channel α gives the model too many degrees of freedom and
  it ends up over-erasing some channels. Single global α is the sweet spot.
- **5_9 (XSI v2 layerwise k=4 + learnable α)**: final_int6_sw **1.24706** → **+0.01446 vs B4-0**,
  much worse. Step_avg ~417ms (~9% slower) → only 1442 steps. Per-block ortho-against-snapshot
  is too disruptive to recover from even with a learnable α.

### Battle 4 Round 2 winner: 5_7 (1.23188 / dup 1.23161)

### Mechanistic hypothesis (validated by 5_7)
In tied-embedding GPT (`tok_emb` weight = lm_head weight), the residual stream's component along
each token's embedding direction IS the prediction signal. XSI v3 explicitly removes the component
along `x0` (the input embedding) before lm_head, which erases part of the tied-embedding signal that
the model uses for next-token prediction. XSA escapes this trap by orthogonalizing against `v`
(a per-layer learned projection) — a soft constraint the layer learns to use productively. XSI's
fixed-direction orthogonalization is a hard constraint the model can't route around.

**Validated by 5_7**: when α is learnable (init 0), the model finds a sweet spot where some XSI
helps but full XSI hurts. The 0.00949 BPB swing between 5_1 (rigid α=1) and 5_7 (learnable α)
is the model dynamically choosing how much input-direction signal to drop. **General principle
worth memorizing**: when adding a destructive op (subtraction, mask, drop), gate it behind a
tanh- or sigmoid-bounded learnable scalar init at zero. Cost: 1 param. Benefit: model opts in
only when helpful, and you preserve the option to recover the baseline at α=0.

### What to try next
- **Learnable scale (5_7)** is the obvious recovery. If even at α≈0 it improves over baseline noise,
  the technique has room. If not, parameterless XSI is dead in this baseline.
- **Apply XSI only to the LAST block** (not all 11) to preserve layer-by-layer information flow.
- **Replace XSA with XSI** in some layers (not adding XSI on top) to see if XSI is a strict
  alternative. Heuron rules allow disabling baseline tricks for our own.
- **Per-channel learnable α** — TESTED (5_8): worse than scalar α. Don't repeat.
- **XSI applied to each layer's `attn_norm(x_in)`** before the attention sees it (rather than to
  attn_out after). Would force the attention's queries/keys/values to be computed from the
  input-orthogonal residual, but preserves the residual stream's direct flow.
- **5_10 (per-block α stack, in-flight)**: 11 per-block scalars + 1 final scalar, all init 0,
  tanh-bounded, applied at every block exit against x0. Strict superset of 5_7.


## Battle 4 Round 3: Cross-sublayer routing (Session 6 cont'd)

**Current record candidate**: 5_21 = **1.22803 BPB** (−0.00385 vs 5_7's 1.23188).
Stack: 5_7 (XSI v3 + scalar α) + 5_15 cross-sublayer MLP gate + per-head attn-output gate.

### KEY PRINCIPLE: Cross-sublayer routing
Lightweight gates (12-input → 1- or per-head sigmoid, zero-init → 1.0) are sensitive to *which* signal
they read from. The Battle 1 winners read from the **static residual** `x_out[..,:12]`. In the new
heuron baseline (Battle 4) those same gates **regress** (5_13: +0.00051). The fix is to read from
the **fresh sub-layer output**: `attn_out[..,:12]` (the freshly computed attention result, not the
residual). 5_15 = 1.22817, the biggest single win of the session.

**Why**: the residual stream `x_out` is information-saturated (skip-connected, EMA-smoothed, etc.),
so a gate reading from it sees a slow-moving signal. `attn_out` is a high-frequency, layer-specific
delta — it carries fresh routing information ("did this attention layer find anything?") which is
exactly what an MLP gate needs to decide whether to amplify or attenuate.

**Generalized rule**: when adding a lightweight gate, prefer reading from the *most recent
sub-layer's raw output* over the residual stream. The gate then routes based on actual sub-layer
activity, not aggregated state.

### Saturation point
- 5_15 (cross-sublayer mlp gate, 1×12): 1.22817
- 5_19 (5_15 + final_norm projection target): 1.22871 — ties
- 5_20 (5_15 with cat(x_in,attn_out) → 1×24 gate): 1.22830 — ties
- 5_21 (5_15 + per-head attn-output gate from x[..,:12]): **1.22803** — strict win
The cross-sublayer MLP gate alone saturates around 1.2283; pairing it with an *orthogonal* gate
(per-head attn output, reading from a different signal) breaks past saturation.

### Things confirmed dead-on-arrival in Battle 4
- Per-channel α (D=512) for XSI: 5_8 = 1.23423, 5_17 = 1.23440 (re-test). Don't try a third time.
- Mid-network XSI insertion: 5_18 = 1.24033. Extra XSI sites are harmful.
- Old Battle 1 gates as-is (read from x_out): 5_13 = 1.23239. Need cross-sublayer reformulation.
- NOV (No Value Projection, V=X) on last 4 layers: 3 attempts at 5_14, all crashed with
  InductorError SMEM 221k > 167k. The dynamo backward kernel for the "expand-broadcast K and use V=X"
  pattern fuses past A100 SMEM. Need to either disable compile on those layers entirely
  (likely incompatible with `fullgraph=True`) or restructure to keep K/V banked but skip the linear.
- XSI applied AFTER final_norm: 5_12 = 1.23312. Order matters; current 5_7 placement is right.
- Gated bigram: 5_16 = 1.23376. Bigram is already strong as-is; gating hurts.

## Battle 4 R3 (cont'd): scaling the per-head attn gate

**Current record candidate**: **5_25 = 1.22661 BPB** (-0.00527 vs 5_7's 1.23188).
Stack: 5_7 (XSI v3 + α) + 5_15 cross-sublayer MLP gate + per-head attn-output gate widened to 48 input dims.

### Width scaling of `attn_gate_w` (per-head, reads `x[..,:W]`)
| W | Branch | BPB | Δ vs prev |
|---|--------|-----|-----------|
| 12 | 5_21 | 1.22803 | — |
| 24 | 5_24 | 1.22782 | −0.00021 |
| 48 | 5_25 | 1.22661 | −0.00121 |
| 64,96,128 | 5_28/29/30 | in flight | scaling sweep |

**Insight**: the per-head attn output gate (Battle 1 pattern, but as a *secondary* lever stacked
on top of cross-sublayer MLP gate) responds *monotonically* to wider input. This is unique — most
other widening attempts (5_20 wide-mlp-gate, 5_26 wide-mlp-gate-on-5_24) ties or regresses. The
attn output gate is a special lever: it gates the attention's contribution per-head per-token,
which has more degrees of freedom to leverage extra input information than the scalar MLP gate.

**Param cost of widening**: 8 heads × W × 11 layers. W=48 → 4224 params. W=128 → 11264 params.
All in scalar group, so they go through Adam (not Muon), zero-init → identity at step 0.

### Things confirmed dead in this scaling round
- Widening cross-MLP gate (5_20: 1×24 cat input, 5_26: same on 5_24): no help, sometimes regression.
- Stacking smear_input_w (5_22) or classic mlp_gate from x_out (5_23) on top: both regress.
- Adding qgain modulation gate (5_27): crashed on state_dict mismatch — needs careful key naming
  or just registering the param in `_HessianAttn` differently.
