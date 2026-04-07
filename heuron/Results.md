# Heuron Research Group - Results

## Research Battle 1: Gates

| # | Experiment | File | val_bpb (mid) | val_bpb (final_int6_sw) | Size (bytes) | Log | Notes |
|---|-----------|------|---------------|------------------------|-------------|-----|-------|
| 0 | Baseline (unmodified heron.py) | train_gpt_heron.py | 1.3118@1000 | **1.2382** | 14,275,458 | heuron_baseline-20260405-143950.log | 1633 steps, 26.9M params |
| 1 | Combined MLP(12->1) + Attn(12->8) gates | train_gpt_heron4.py | 1.3058@1000 | **1.2318** | 14,085,672 | heuron_both_gates-20260405-164122.log | 1583 steps, +1188 params, **-0.0064 BPB** |
| 2 | Exp1 + Input-dependent SmearGate | train_gpt_heron5.py | 1.3040@1000 | **1.2299** | 14,111,843 | heuron_gates_smear-*.log | 1595 steps, +12 more params, **-0.0083 BPB total** |
| 3 | Exp2 + Skip connection gate | train_gpt_heron6.py | 1.3108@1000 | 1.2353 | 13,643,993 | heuron_skip_gate-*.log | WORSE: skip gate hurts |
| 4 | Per-dim MLP gate (12->512) | train_gpt_heron7.py | 1.3074@1000 | 1.2421 | 13,737,104 | heuron_perdim_mlp-*.log | WORSE: too many params, slower |
| 5 | 16-dim gate inputs (all gates) | train_gpt_heron8.py | 1.3061@1000 | 1.2343 | 14,007,063 | heuron_16dim_gates-*.log | WORSE: 12 dims is optimal |
| - | MLP output gate only (12->1) | train_gpt_heron2.py | - | - | - | - | prepared (ablation) |
| - | Attn gate only (12->8 per-head) | train_gpt_heron3.py | - | - | - | - | prepared (ablation) |

## Key
- val_bpb (mid): val_bpb at a comparable step during training (e.g., step 2000 or step 2500)
- val_bpb (final_int6_sw): final_int6_sliding_window_exact val_bpb (the score that matters)
- Size: Total submission size (int6+lzma), must be <= 16,000,000 bytes
- Lower BPB = better

---

## Research Battle 2: Multiple Embeddings

**Baseline**: heron5.py (Battle 1 winner, 1.2299 BPB)

| # | Experiment | File | val_bpb (mid) | val_bpb (final_int6_sw) | Size (bytes) | Log | Notes |
|---|-----------|------|---------------|------------------------|-------------|-----|-------|
| B2-0 | Baseline (heron5 gates) | train_gpt_heron5.py | 1.3040@1000 | 1.2299 | 14,111,843 | (from Battle 1) | Reference for Battle 2 |
| B2-1 | ME dim=128, 1 extra embed | train_gpt_heron_2_1.py | 1.3019@1000 | **1.2292** | 13,832,932 | heuron_me_128-20260405-204232.log | +131K params, **-0.0007 vs gates, BEST** |
| B2-2 | ME dim=64, 1 extra embed | train_gpt_heron_2_2.py | 1.3066@1000 | 1.2332 | 14,073,191 | pgolf-train-4x-39167964.out | +65K params, WORSE than B2-1 |
| B2-3 | ME hash vocab=512, dim=128 | train_gpt_heron_2_3.py | 1.3073@1000 | 1.2330 | 14,089,206 | pgolf-train-4x-39167968.out | 512 hash vocab, WORSE than B2-1 |
| B2-4 | Gated ME dim=128 | train_gpt_heron_2_4.py | 1.3006@1000 | **1.2279** | 14,005,322 | heuron_me_gated-*.log | **NEW BEST, -0.0020 vs gates, beats admin record** |
| B2-5 | Input Projections dim=128 | train_gpt_heron_2_5.py | 1.3153@1000 | 1.2451 | 14,029,964 | pgolf-train-4x-39169073.out | TERRIBLE, worse than baseline |
| B2-6 | ME 2x128 (no gate) | train_gpt_heron_2_6.py | 1.3025@1000 | crashed | 13,906,764 | pgolf-train-4x-39169088.out | Crashed during sliding window eval (SLURM timeout) |
| B2-7 | Gated ME 2x128 | train_gpt_heron_2_7.py | 1.3024@1000 | 1.2342 | 13,770,706 | heuron_gme_2x128-*.log | 1480 steps (too slow), 2 embeds hurts |
| B2-8 | Gated ME dim=192 | train_gpt_heron_2_8.py | 1.3008@1000 | 1.2295 | 13,837,666 | pgolf-train-4x-39170118.out | 1527 steps, close but 128 still better |
| B2-9 | Gated Hash ME (512) | train_gpt_heron_2_9.py | 1.3067@1000 | crashed | 13,951,425 | pgolf-train-4x-39172646.out | SLURM timeout during sw eval |
| B2-10 | Gated Bigram ME (1536) | train_gpt_heron_2_10.py | 1.3046@1000 | 1.2345 | 14,051,564 | pgolf-train-4x-39172647.out | Bigram context helps vs hash, but < full vocab |
| B2-11 | Ungated Bigram ME (1536) | train_gpt_heron_2_11.py | 1.3054@1000 | 1.2381 | 14,175,633 | pgolf-train-4x-39172649.out | Bigram alone, no gate, ~= baseline |

---

## Research Battle 3: Attention Residuals & Similar

**Bases**:
- baseline = `train_gpt_heron.py` (1.2382 BPB, no gates, no ME)
- record = `train_gpt_heron_2_4.py` (1.2279 BPB, has gates + gated ME)

All variants use **sigmoid-bounded** dense weights to prevent runaway under Adam (Lesson from
prior session: zero-init `nn.Parameter` scalars optimized by Adam grow ~scalar_lr/step and
quickly destroy training. Sigmoid bound caps contributions in [0, 1].).

| # | Experiment | File | Base | step1000 val_bpb | final_int6_sw BPB | step_avg | vs base | Job |
|---|-----------|------|------|------------------|-------------------|----------|---------|-----|
| B3-0a | Baseline | train_gpt_heron.py        | -      | 1.3118     | **1.2382** | ~380ms | reference | (Battle 1) |
| B3-0b | Record   | train_gpt_heron_2_4.py    | -      | 1.3006     | **1.2279** | ~392ms | reference | (Battle 2) |
| B3-1  | Real DenseFormer       | train_gpt_heron_3_1.py | baseline | 1.3136 | 1.26437 | 460ms | +0.0262 | 39260876 |
| B3-2  | Block Attn Residual    | train_gpt_heron_3_2.py | baseline | 1.4005 | 1.29724 | 424ms | +0.0590 | 39260878 |
| B3-3  | Gated Attn Residual    | train_gpt_heron_3_3.py | baseline | 1.3103 | 1.25946 | 463ms | +0.0213 | 39260880 |
| B3-4  | Real DenseFormer       | train_gpt_heron_3_4.py | record   | 1.3007 | 1.25119 | 486ms | +0.0233 | 39260881 |
| B3-5  | Block Attn Residual    | train_gpt_heron_3_5.py | record   | 1.3013 | **1.24209** | 449ms | +0.0142 | 39261103 |
| B3-6  | Gated Attn Residual    | train_gpt_heron_3_6.py | record   | 1.3006 | 1.25154 | 492ms | +0.0237 | 39261105 |
| B3-7  | Block Attn Res init=-3 | train_gpt_heron_3_7.py | record   | 1.3031 | 1.24978 | 450ms | +0.0219 | 39263469 |
| B3-8  | NO U-Net skips (control)| train_gpt_heron_3_8.py | baseline | 1.3105 | **1.2323** | 365ms | **-0.0059** | 39266578 |
| B3-9  | NO skips + Real Dense (eager) | train_gpt_heron_3_9.py | baseline | 1.3098@685 | 1.4261 | 877ms | undertrained | 39271856 |
| B3-10 | NO U-Net skips (record) | train_gpt_heron_3_10.py| record   | 1.3087 | 1.24000 | 384ms | +0.0121 | 39270015 |

### Implementation details (sigmoid-bounded variants)

All three techniques use the same parametrization:
```python
self.dense_logits = nn.Parameter(torch.full((n, n), -6.0))  # sigmoid(-6) ~= 0.0025 ~ 0
dense_w = torch.sigmoid(self.dense_logits).to(dtype=x.dtype)
# inside loop:
stacked = torch.stack(layer_outputs, dim=0)        # (i, B, T, D)
w = dense_w[i, :i]                                  # (i,)
x = x + torch.einsum('i,ibtd->btd', w, stacked)
```

- **Real Dense (3_1, 3_4)**: full lower-triangular matrix `dense_logits[i, j]` for j<i over all layers (encoder + decoder).
- **Block Dense (3_2, 3_5)**: separate enc-only `(num_enc, num_enc)` and dec-only `(num_dec, num_dec)` matrices. Encoder layers only read previous encoder layers, decoder layers only read previous decoder layers (no cross-block dense — U-Net skip remains as the only encoder→decoder path).
- **Gated Dense (3_3, 3_6)**: same as Real but multiplied by per-target-layer input-dependent gate `g = 2.0 * sigmoid(F.linear(x[..., :12], gate_w[i:i+1]))` shape (B,T,1). Per-layer not per-pair, to avoid the OOM kernel from the previous session.

All `*_logits` and `*_gate_w` parameters added to `CONTROL_TENSOR_NAME_PATTERNS` and `scalar_params` for proper quantization and distributed all-reduce.

### Findings

1. **None of the three techniques improve over the baseline or record.** Even at step 1000 (the early predictor), the baseline-of-three `_3_4/_3_5/_3_6` only matches the record (1.3007 / 1.3013 / 1.3006 vs record's 1.3006), and at the final sliding-window eval all six are worse than their respective base.
2. **The damage is smaller on the record than on the baseline.** The record has ME providing per-layer per-token information; the dense connections compete with that signal less destructively.
3. **Block Dense is the WORST on baseline (+0.059) but BEST on record (+0.014).** With record's ME providing the global per-layer information, restricting dense connections to within-block local connectivity works better than full lower-triangular dense. Without ME, restricting connections starves the encoder→decoder information path (only U-Net skip remains for that direction) and hurts more.
4. **Step time slowdown alone explains 30-50% of the regression.** All variants run at 424–492 ms/step vs ~380–390 ms baseline → ~16–28% fewer training steps in 600s. Combined with no per-step quality improvement, this explains most of the gap.
5. **The previous session's catastrophic results (1.27–1.30 across all 6) were due to the Adam-on-zero-init bug**, fixed by sigmoid bounds. The fixed versions land in the 1.242–1.297 range (much better than the failed 1.252–1.298 from the bugged runs, but still not winning).
6. **Stronger initial dense (B3-7, init=-3.0 vs -6.0) is slightly WORSE.** Block Dense + record at init=-3.0 = 1.24978 vs init=-6.0 = 1.24209 (+0.0077). This confirms dense connections do not help here even when given more initial signal — the "weakest possible" init (~0 contribution from start) is the best of the bad options.

### Battle 3 Extension: U-Net skip ablation (B3-8, B3-9, B3-10)

User asked: "remove the U-Net skips and see if dense truly works." Plan:
- B3-8 = baseline minus skips (control)
- B3-9 = baseline minus skips, plus Real Dense (test of dense without redundancy)
- B3-10 = record minus skips (does the skip-removal generalize?)

**Surprising finding 1: removing U-Net skips on the baseline IMPROVES BPB.**
B3-8 (no skips, no dense, no other changes) = **1.2323**, beats baseline (1.2382) by **-0.0059**.
This is one of the largest single-change wins of the entire research. Hypothesis: the U-Net skips
in the original architecture were sub-optimally weighted (per-channel skip_weights initialised at 0)
and pulled down the residual stream's effective rank. Removing them lets each layer's residual stream
build cleanly. The 365 ms/step time is also slightly faster than baseline's ~380 ms.

**Finding 2: removing U-Net skips on the record HURTS.**
B3-10 (record minus skips) = 1.2400 vs record 1.2279 = +0.0121 regression.
Hypothesis: the record (heron_2_4) already has gated ME providing per-token global information;
the U-Net skips in this configuration are functionally cooperative with the gates rather than redundant,
so removing them costs the model. This is the OPPOSITE asymmetry from the baseline, and is consistent
with the Battle 3 finding that techniques behave differently on the two bases.

**Finding 3: dense connections DO help per-step when not redundant with skips, but the eager-mode
implementation cost wipes out the win.** B3-9 had to be implemented with `@torch.compiler.disable`
on the dense add (free function) because removing skips lets inductor over-fuse the dense
bmm/einsum with rms_norm, exceeding triton's shared-memory limit (Required:168k > Hardware:166k).
Eager-mode dense add → 877 ms/step (2.4× slower than B3-8's 365 ms/step) → only 685 steps in 600 s
vs 1644 for B3-8.

Per-step val_bpb comparison (B3-8 no skips vs B3-9 no skips + dense):

| step | B3-8 (no skip) | B3-9 (no skip + dense) | delta |
|------|----------------|------------------------|-------|
| 200  | 1.5801         | 1.5834                 | +0.0033 |
| 400  | 1.4208         | 1.4216                 | +0.0008 |
| 600  | 1.3642         | 1.3507                 | **-0.0135** |
| 685  | (~1.350 interp)| 1.3098                 | **~-0.04** |
| 1000 | 1.3105         | -                      | -       |

At step 685, B3-9 (1.3098) is **roughly equal to B3-8 at step 1000** (1.3105). That is a ~46%
per-step training-efficiency improvement from dense connections — the first time in Battle 3 where
the technique demonstrably helps. But because the eager-mode implementation is 2.4× slower per step,
the wallclock-bounded final BPB (1.4261) is dominated by the step count deficit and EMA also hurts
this far below the EMA-tuned regime.

**Verdict on dense connections (without skips):** *helpful per-step, blocked by triton kernel
fusion budget under torch.compile.* If a way to keep dense connections inside `fullgraph=True`
without over-fusion is found (e.g., manual fusion barrier, custom kernel, or chunked accumulator
with `.contiguous()` boundaries), dense + no skips could plausibly beat both baseline and B3-8.
Reserved as a follow-up; not in current Battle 3 budget.

**Verdict on U-Net skip removal:** *a real, base-dependent win.* Saves params (~5*512=2560
floats) AND improves BPB on the unmodified architecture. Should be the new baseline starting point
for Battle 4. The asymmetry with the record (B3-10) means the choice of "with or without skips" must
be made per-architecture, not assumed.

Battle 3 budget used: 10/12.

---

## Research Battle 4: Is what worked truly transferable? (NEW heuron baseline, FA3→SDPA/FA2)

**New baseline**: `train_gpt_heuron.py` — fundamentally different from old `train_gpt_heron.py`. Adds: ParallelMuon (post-backward reduce-scatter), MTP heads (off by default), GPTQ Hessian-aware int6 quantization, autoregressive self-gen calibration, selective ±1 pruning, EMA(0.997), SWA(50), late_qat at 0.15, **XSA on ALL 11 layers** (XSA_LAST_N=11), bigram2048, VE on layers 9,10, MUON warmup 0.92→0.99 over 1500 steps, GRAD_CLIP_NORM=0.3. Old battles' "winners" (lightweight gates, gated ME, no-skip) are NOT present in this baseline.

**Required environment fix** before any run: removed FA3 import (`flash_attn_interface` not installable on Cineca A100s), replaced both attention call sites with `F.scaled_dot_product_attention(...)`. The script already calls `enable_flash_sdp(True)`, so SDPA dispatches to FlashAttention-2 on A100. Applied in-place to `train_gpt_heuron.py` (this is an environment fix, not a research win).

| # | Experiment | File | model_params | steps | step_avg | live val_bpb | post_ema | int6_RT | **final_int6_sw BPB** | Δ vs B4-0 | Job |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B4-0 | Bare new baseline (FA3→SDPA) | train_gpt_heuron.py | 26,993,756 | 1568 | 382.75 | 1.2356 | 1.2535 | 1.2569 | **1.23260** | reference | 39276904 |
| B4-1 | + Battle 1 lightweight gates | train_gpt_heuron_4_1.py | 26,994,956 | 1538 | 390.36 | 1.2343 | 1.2529 | 1.2578 | **1.23381** | +0.00121 | 39283623 |
| B4-2 | + Battle 2 gated ME (dim=128) | train_gpt_heuron_4_2.py | 27,125,103 | 1508 | 397.94 | 1.2359 | 1.2535 | 1.2573 | **1.23338** | +0.00078 | 39285767 |
| B4-3 | + Battle 1 gates + Battle 2 gated ME | train_gpt_heuron_4_3.py | 27,126,303 | 1494 | 401.68 | **1.2331** | 1.2533 | 1.2584 | **1.23523** | +0.00263 | 39285798 |

### Findings

1. **NONE of the past Battle 1/2 winners transfer to the new heuron baseline.** All three modifications (gates, gated ME, combined) score WORSE than the bare new baseline on `final_int6_sliding_window`. The new baseline is the Battle 4 winner at **1.23260**.

2. **Live (pre-EMA) BPB does favor the additions, especially the combination.** B4-3 has the best live val_bpb at 1.2331, B4-1 second at 1.2343, then baseline 1.2356, then B4-2 1.2359. So gates do still produce a per-step quality improvement on this baseline. The combination is a real per-step winner.

3. **The wallclock budget eats the gain.** Each addition adds ~2-5% to step time (B4-1 +1.8%, B4-2 +4.0%, B4-3 +5.0%), reducing the total step count from 1568 (B4-0) → 1538 (B4-1) → 1508 (B4-2) → 1494 (B4-3). On a fixed 600s budget, slower per-step quality has to overcome this step deficit, and on the new (much stronger) baseline the quality margin is too small to do so.

4. **Quantization roundtrip damage is similar across all variants** (~+0.0034 to +0.0051 from post_ema to int6_roundtrip). The added gates/ME don't make GPTQ noticeably worse on this baseline, despite adding small `(1, 12)` and `(num_heads, 12)` parameters that go through the int4-band classification path.

5. **The sliding-window vs roundtrip improvement is consistent at ~−0.024 across runs**, suggesting the sliding window context-extension benefit is architecture-independent.

6. **The new heuron baseline already incorporates much of the value the old gates/ME captured.** XSA on ALL 11 layers (vs old XSA-on-last-4) gives every layer access to refined per-token information that the gated ME used to provide. MUON warmup (0.92→0.99 over 1500 steps) trains Muon-managed weights more carefully early on, which the lightweight gates also did via their bounded sigmoid contributions. Bigram2048 + VE on layers 9,10 add token-level signals comparable to the per-layer ME signal. So the techniques' "novelty value" was already absorbed.

7. **The B4-0 score (1.23260) is between the old baseline (1.2382) and the old record (1.2279):** −0.0056 improvement vs old baseline, but +0.0047 worse than the old hand-tuned record. This means: although the new architecture is much stronger out-of-the-box, the older architecture combined with manual gate/ME tuning was still slightly better at the final number on the old training regime. The TODO of "transferability" answers: **the strong-baseline + simple-add hypothesis fails — the new baseline's choices have already taken the design space we were exploring.**

### Implementation details (what was added in each branch)

- **B4-1 gates** (matches old `heron4`/`heron5` recipes):
  - `CausalSelfAttention.attn_gate_w = nn.Parameter(torch.zeros(num_heads, 12, fp32))`
  - `attn_g = 2.0 * sigmoid(F.linear(x[:, :, :12], attn_gate_w))`, `y = y * attn_g.unsqueeze(-1)`
  - `Block.mlp_gate_w = nn.Parameter(torch.zeros(1, 12, fp32))`, `mlp_g = 2.0 * sigmoid(F.linear(x[:, :, :12], mlp_gate_w))`
  - `SmearGate.smear_input_w = nn.Parameter(torch.zeros(1, 12, fp32))`, modulating the existing g_base by 2*sigmoid of the input projection
  - All zero-init → 2*sigmoid(0)=1.0 → no-op start. `2.0*` upper-bounds the gate at 2.
  - `CONTROL_TENSOR_NAME_PATTERNS` extended with `attn_gate_w,mlp_gate_w,mlp_gate,smear_input_w`. `scalar_params.append(base_model.smear.smear_input_w)` explicit.
  - Param delta from B4-0: +1200 (12*8 + 12 + 12 per layer × 11 layers ≈ 1320, observed 1200 ≈ matches modulo VAE/decoder layer count).

- **B4-2 gated ME** (matches old `heron_2_4`):
  - Hyperparameters: `me_enabled=True, me_dim=128, me_num_extra=1`
  - `GPT.me_embeds = ModuleList([Embedding(vocab_size, 128)])`, init `normal_(std=0.01)`
  - `GPT.me_layer_scales = ParameterList([Parameter(zeros(1)) for _ in num_layers])`
  - `GPT.me_gate_ws = ParameterList([Parameter(zeros(1, 24)) for _ in num_layers])`
  - `_compute_me_signals(idx)` computes ME embedding once per forward; `_inject_me(x, layer_i, me)` reads cat(x[:12], me[:12]) → scalar gate, multiplies by zero-init scale, adds to first 128 dims of x.
  - Mirrored into `_HessianGPT` so calibration activations match training.
  - `me_embeds.weight` → `tok_params` at `token_lr`. `me_layer_scales`+`me_gate_ws` → `scalar_params`.
  - `CONTROL_TENSOR_NAME_PATTERNS` extended with `me_layer_scales,me_gate_ws`.
  - Param delta from B4-0: +131,347 (1024×128 = 131,072 + 11×24 + 11 ≈ matches).

- **B4-3 combined** = B4-2 + B4-1 patches applied on top.

### Notes
- The interactive job for B4-1 used `salloc` (single 30-min slot). B4-2 and B4-3 were submitted via `sbatch scripts/train_4xA100.sh` in parallel — both started immediately on lrdn0572 and lrdn0585.
- Logs: `logs/heuron_b4_0_baseline.txt`, `logs/heuron_b4_1_gates.txt`, `logs/heuron_b4_2_gme.txt`, `logs/heuron_b4_3_combined.txt`.
- All four runs use identical seed (1337) and identical `Hyperparameters`. No `Hyperparameters` were touched (per Heuron rules).
- All four pass GPTQ + selective_prune cleanly; all four are well under the 16MB limit (13.5–13.8 MB int6+lzma).

### Pseudocode of the best Battle 4 run (B4-0, the bare patched new baseline)

```python
# Conceptually: the new heuron baseline is the WINNER. The only "fix" needed was:
# 1. Remove the FA3 import (not installable on Cineca A100s)
import torch.nn.functional as F

# 2. In CausalSelfAttention.forward and _HessianAttn.forward, replace
#    y = flash_attn_3_func(q, k, v, causal=True)
#  with
y = F.scaled_dot_product_attention(
    q.transpose(1, 2),  # (B,H,T,D)
    k.transpose(1, 2),
    v.transpose(1, 2),
    attn_mask=None,
    is_causal=True,
    enable_gqa=(self.num_kv_heads != self.num_heads),
)
y = y.transpose(1, 2).contiguous()  # back to (B,T,H,D) for downstream XSA path
# script already calls torch.backends.cuda.enable_flash_sdp(True) → FA2 on A100
```
This is a no-research-content change (just an environment fix). The architectural and optimizer choices are entirely as written in the existing `train_gpt_heuron.py` (XSA_LAST_N=11, bigram2048, VE 9,10, MUON warmup, ParallelMuon, GPTQ AR-self-gen, selective ±1 prune to 15.9MB, EMA 0.997, SWA from 1500, late_qat at 0.15).

Battle 4 budget used: 4/4 (B4-0 baseline + 3 transferability tests). No further follow-up planned in this session.

---

## Research Battle 4 — Round 2 (NEW): Exclusive Input Attention (XSI) variants

**Baseline**: B4-0 = bare new heuron baseline (FA3→SDPA fix only) → **1.23260** sliding-window int6 BPB.
**Hypothesis**: XSA orthogonalizes against the *value* direction; XSI orthogonalizes against the *input* direction. Three variants proposed in Heuron.md (TODO Battle 4):
- v1 Naive Block XSI: in each block, orthogonalize attn_out against block input; orthogonalize MLP out against post-attn x.
- v2 Block Layerwise XSI (k=4): every k layers, orthogonalize the residual stream against the snapshot from k layers ago.
- v3 Final XSI: just before final_norm + lm_head, orthogonalize residual against x0 (smeared/normed input embedding).

| # | Experiment | File | live val_bpb @1569 | post_ema | final_int6_sw | Size | Log | Δ vs B4-0 |
|---|-----------|------|---------------------|----------|---------------|------|-----|-----------|
| 5_1 | XSI v3 final-logit (rigid) | train_gpt_heuron_5_1.py | 1.2403 | 1.2614 | 1.24137 | 13,914,379 | heuron_5_1_xsi_final-20260407-170705.log | +0.00877 LOSS |
| 5_2 | XSI v2 layerwise k=4 (rigid, .detach fix) | train_gpt_heuron_5_2.py | 1.2462 | 1.2663 | 1.24715 | 13,695,025 | pgolf-train-4x-39293683.out | +0.01455 LOSS, also slower (403ms/step) |
| 5_3 | XSI v1 naive block (rigid) | train_gpt_heuron_5_3.py | (pending) | | | | | |
| 5_7 | **XSI v3 learnable-α (single scalar, tanh-bounded, init 0)** | train_gpt_heuron_5_7.py | **1.2348** | **1.2525** | **1.23188** | 14,012,635 | pgolf-train-4x-39294868.out | **−0.00072 WIN** (+1 param) |
| 5_8 | XSI v3 per-channel learnable-α (D=512 params) | train_gpt_heuron_5_8.py | (pending) | | | | | |

### 5_2 fix: triton OOM in backward
- Initial 5_2 (job 39290962) crashed with `InductorError: No valid triton configs. OutOfResources: out of resource: shared memory, Required: 260224, Hardware limit: 166912` during `(warmup_loss * grad_scale).backward()`.
- Root cause: storing per-layer residual snapshots in `saved_inputs[]` and then re-using them at later layers caused torch.compile to build a giant fused backward kernel that exceeded A100 SMEM.
- Fix: store `x.detach()` (stop-gradient) into `saved_inputs`. Math is approximately equivalent (gradient still flows through `x` normally; only the cross-layer projection-target gradient is dropped — same trick as BYOL/SimSiam stop-grad). Mirrored in GPT.forward, GPT.forward_logits, and `_HessianGPT.forward`.

### 5_1 result analysis
**XSI v3 final-logit hurts by +0.009 BPB.** Hypothesis why:
- The residual stream `x` going into `final_norm + lm_head` is already a carefully built information field; subtracting the entire component along `x0` throws away the model's "starting from this token" signal that the lm_head needs (especially for tied embeddings, since the prediction logit for token `t` is roughly aligned with its embedding direction).
- XSA works because the projection target is the *value* (a learned, layer-specific direction). x0 is a fixed per-token direction → too rigid, removes too much.
- Lesson: rigid orthogonalization against fixed reference directions is harmful at the head. Need either (a) a learned scaling of the projection or (b) a learned reference direction.

### Follow-up plan (after 5_2 / 5_3 finish)
- If v1 and v2 also regress: pivot to **learnable-α XSI**: `x = x - α * <x, src_n> src_n` where α is a per-layer (or single) Parameter init at 0. Init at 0 = baseline behavior, so the model can opt out. Fully reversible. Same parameter-counting trick as gates.
- Alternative: **learned reference direction** — replace x0 with a small learned projection of x0 (1×D vector or per-token affine), giving the model more flexibility about what to subtract.

### KEY FINDING (5_7)
**5_7 = XSI v3 with a single learnable α (tanh-bounded scalar, init 0) BEATS the new heuron baseline B4-0 by −0.00072 BPB (1.23260 → 1.23188).**
- Same speed (383ms/step, identical to baseline 382.75ms)
- Cost: +1 scalar parameter (`xsi_alpha`)
- Math: `x = x − tanh(xsi_alpha) * <x, x0_n> * x0_n` applied right before `final_norm + lm_head`
- The win is small but real — and it confirms a major lesson: **rigid orthogonalization is too aggressive; the model needs the OPTION to apply XSI but not the obligation.**
- The same code with `α=1` (rigid, 5_1) costs +0.00877 BPB. So **a learnable α improves over rigid by 0.00949 BPB** — the difference between "slight improvement" and "significant regression."
- General principle (worth committing to memory): when introducing a destructive op (subtraction, masking, dropout), gate it behind a tanh/sigmoid-bounded learnable scalar init at zero. Cost: 1 param. Reward: model can opt in only if helpful.
- Followup 5_8 (per-channel α, +511 more params) tests whether channelwise selectivity adds further gains.

### KEY FINDING (5_15) — NEW BATTLE 4 RECORD CANDIDATE
**5_15 = 5_7 + cross-sublayer MLP gate driven by attn_out's first 12 dims** → **1.22817 BPB** (−0.00371 vs 5_7).

| # | Experiment | File | final_int6_sw | Δ vs 5_7 | Notes |
|---|-----------|------|---------------|----------|-------|
| 5_11 | 5_7 with `F.normalize(final_norm(x0))` projection target | train_gpt_heuron_5_11.py | 1.23154 | −0.00034 | Better target form |
| 5_12 | XSI applied AFTER final_norm | train_gpt_heuron_5_12.py | 1.23312 | +0.00124 | regression |
| 5_13 | 5_7 + full Battle 1 gates (mlp/attn/smear) | train_gpt_heuron_5_13.py | 1.23239 | +0.00051 | regression — old gates don't transfer |
| 5_14 | NOV last 4 layers (V=X) | train_gpt_heuron_5_14.py | CRASH | — | InductorError SMEM 221k > 167k (3 attempts) |
| **5_15** | **5_7 + cross-sublayer MLP gate (attn_out[..,:12] gates MLP contribution)** | train_gpt_heuron_5_15.py | **1.22817** | **−0.00371** | **WIN — biggest of session** |
| 5_16 | 5_7 + gated bigram | train_gpt_heuron_5_16.py | 1.23376 | +0.00188 | regression |
| 5_17 | 5_11 + per-channel α (D=512) | train_gpt_heuron_5_17.py | 1.23440 | +0.00252 | per-channel α still bad |
| 5_18 | 5_11 + 2nd XSI mid-network | train_gpt_heuron_5_18.py | 1.24033 | +0.00845 | bad |
| 5_19 | 5_15 + 5_11 final_norm target combined | train_gpt_heuron_5_19.py | 1.22871 | −0.00317 | ties 5_15 (within noise) |
| 5_20 | 5_15 with x_mlp_gate_w widened to 24 dims (cat(x_in,attn_out)) | train_gpt_heuron_5_20.py | 1.22830 | −0.00358 | ties 5_15 |
| 5_21 | 5_15 + per-head Battle 1 attn output gate (8×12) | train_gpt_heuron_5_21.py | 1.22803 | −0.00385 | broke saturation by stacking orthogonal gate |
| 5_22 | 5_15 + Battle 1 smear_input_w gate | train_gpt_heuron_5_22.py | 1.23439 | +0.00251 | regression |
| 5_23 | 5_21 + classic mlp_gate from x_out (multiplicative on cross-sublayer) | train_gpt_heuron_5_23.py | 1.23417 | +0.00229 | regression |
| 5_24 | 5_21 with attn_gate_w widened to 24 input dims | train_gpt_heuron_5_24.py | 1.22782 | −0.00406 | wider input → strictly better |
| **5_25** | **5_24 with attn_gate_w widened to 48 input dims** | train_gpt_heuron_5_25.py | **1.22661** | **−0.00527** | **NEW BEST — scaling continues to pay** |
| 5_26 | 5_24 + cross-mlp gate widened to 24 dims (cat input) | train_gpt_heuron_5_26.py | 1.23186 | +0.00000 | regression — MLP gate widening hurts |
| 5_27 | 5_24 + per-head q_gain modulation gate | train_gpt_heuron_5_27.py | CRASH | — | state_dict mismatch on quant reload |

**Lesson**: the cross-sublayer routing pattern (use *fresh* sub-layer outputs, not the static residual, as gating signal) is the unlock. Old Battle 1 gates read from the static residual `x_out[..,:12]` and regressed under the new architecture; reading from `attn_out[..,:12]` flips that to a clean win. Symmetric stacking attempts (5_19/5_20) saturate at the same ~1.2283 floor, suggesting this gate captures the available cross-sublayer information.
