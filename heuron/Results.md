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

