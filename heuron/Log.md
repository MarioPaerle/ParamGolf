# Heuron Research Group - Log

## Research Battle 3: Attention Residuals & Similar

### 2026-04-05 - Session 1 Start (interrupted)
- Read all Battle 1 & 2 artifacts (Memory, Results, Scratch, Recap)
- Understood architecture: 11 layers, 512 dim, 1024 vocab, UNet encoder(5)/decoder(6)
- Battle 2 best (heron_2_4.py): 1.2279 BPB (gated ME + gates)
- Baseline (heron.py): 1.2382 BPB
- Read survey: AttentionResiduals_Survey.md (20+ papers)
- Battle 3 plan: 3 techniques × 2 bases + ablations (max 12 experiments)
  - DenseFormer (additive zero-init per-layer dense connections)
  - Block Attention Residuals (dense within encoder/decoder blocks only)
  - Gated Attention Residuals (dense + input-dependent 12-dim gates)
- Created baseline variants: heron_3_1 (Dense), heron_3_2 (Block), heron_3_3 (Gated)
- **SESSION INTERRUPTED** before any runs

### 2026-04-06 - Session 2 Start
- Verified 3 baseline implementations (correct CONTROL_TENSOR, scalar_params, forward/eval)
- First batch submit (39191226-28): DenseFormer FAILED with InductorError (FusedMixOrderReductions)
  - Root cause: iterative `x = x + da[j] * layer_outputs[j]` creates too-complex backward graph
  - Fix: replaced with `torch.stack` + `torch.einsum` pattern in all 6 files
- Created record-based variants: heron_3_4 (Dense+record), heron_3_5 (Block+record), heron_3_6 (Gated+record)
- Resubmitted 3 baseline jobs (39191578-80): all completed but with bad BPB (1.281, 1.281, 1.298)
- Submitted 3 record jobs (39191591, 39191595-96, 39192427, etc): bad BPB (1.269, 1.252, 1.277)
- **CRITICAL BUG IDENTIFIED**: All 10 prior runs failed because `dense_alphas` was a zero-init scalar
  Parameter optimized by Adam with scalar_lr=0.025. Adam normalizes gradients, so alphas grow ~lr/step,
  reaching ~1.25 after 50 steps. With 11 layers each adding ~1.25 * (layer output), residual stream
  exploded ~10x, destroying training. Plus heron_3_3/3_6 OOMed due to per-pair gates being too complex.

### 2026-04-07 - Session 3 Start
- Read entire codebase from Heuron.md / recap.md / papers
- Confirmed prior agent's diagnosis & fix plan:
  - Replace `dense_alphas = nn.Parameter(zeros(...))` with
    `dense_logits = nn.Parameter(full(..., -6.0))` and use `sigmoid(dense_logits)`.
    sigmoid(-6) ~= 0.0025 (~zero init), bounded in [0,1] so contributions cannot explode.
  - Replace per-pair gates with per-target-layer gates: `dense_gate_w[i:i+1]` shape (1, 12),
    avoiding the OOM kernel.
- Rewrote all 6 heron_3_x files:
  - 3_1: Real DenseFormer (baseline)
  - 3_2: Block Attention Residual (baseline) - separate enc/dec dense logit matrices
  - 3_3: Gated Attention Residual (baseline) - sigmoid-bounded weights + per-target gate
  - 3_4: Real DenseFormer (record / heron_2_4 base, with ME)
  - 3_5: Block Attention Residual (record)
  - 3_6: Gated Attention Residual (record)
- All 6 pass `python3 -c "import ast; ast.parse(...)"` syntax check
- Submitted all 6 as parallel sbatch jobs (39260876, 39260878, 39260880, 39260881, 39261103, 39261105)
- All 6 ran successfully on 4xA100, no crashes, no OOM, no compilation errors

### 2026-04-07 - Session 3 Battle 3 Results
| File | Description | step1000 | final BPB | vs base |
|------|-------------|----------|-----------|---------|
| heron_3_1 | Real Dense + baseline | 1.3136 | 1.26437 | +0.0262 |
| heron_3_2 | Block Dense + baseline | 1.4005 | 1.29724 | +0.0590 |
| heron_3_3 | Gated Dense + baseline | 1.3103 | 1.25946 | +0.0213 |
| heron_3_4 | Real Dense + record | 1.3007 | 1.25119 | +0.0233 |
| heron_3_5 | Block Dense + record | 1.3013 | 1.24209 | +0.0142 |
| heron_3_6 | Gated Dense + record | 1.3006 | 1.25154 | +0.0237 |

- **NONE of the 6 attention-residual variants beat their respective base.**
- Best of group: heron_3_5 (Block Dense + record) at 1.24209, still +0.0142 worse than record.
- Block Dense behaves OPPOSITE on the two bases: WORST on baseline (+0.059), BEST on record (+0.014).
  Hypothesis: with record's per-layer ME providing global per-token information, restricting dense
  to within-block local connectivity stops it from competing with ME's signal. On the baseline,
  block dense starves the encoder→decoder cross-block info path (only U-Net skip remains).
- Step time slowdown alone (424–492 ms vs ~380 ms baseline) explains 30–50% of the regression.
  All variants run 1235–1417 steps in 600s vs ~1580 baseline.
- Submitted 1 ablation: heron_3_7 = Block Dense + record with init at -3.0 (sigmoid(-3)≈0.0474,
  ~20× stronger initial contribution) to test if "more aggressive initial dense" helps. Job 39263469.
- heron_3_7 result: 1.24978 (+0.0219 vs record). WORSE than heron_3_5 init=-6 (1.24209).
  Confirms: stronger initial dense hurts. The "weakest possible" init is the least bad.

### Battle 3 Initial Conclusion (2026-04-07, pre-skip-ablation)
- 7 experiments run this session, all with sigmoid-bounded weights (no Adam runaway).
- ZERO of the 7 attention residual variants beat their respective base.
- Best of group: heron_3_5 (Block Dense + record, init=-6) at 1.24209, +0.0142 vs record 1.2279.
- All variants pay 16-28% step time overhead with no per-step quality improvement.
- Conclusion as user predicted: "I don't expect it to beat the baseline nor make record" — true.

### 2026-04-07 - Session 3 Battle 3 Extension: U-Net skip ablation
- User asked: "remove the U-Net skips and see if dense truly works?"
- Plan: B3-8 (no skips control), B3-9 (no skips + dense), B3-10 (no skips on record).
- B3-8 (heron_3_8.py): vanilla baseline minus skips
  - **RESULT: 1.2323** (vs baseline 1.2382 = **-0.0059**, 1646 steps, 365 ms/step)
  - **Surprise**: removing skips alone *improves* the baseline.
  - Hypothesis: 0-init per-channel skip_weights pulled down the residual stream's effective rank.
- B3-10 (heron_3_10.py): record minus skips
  - **RESULT: 1.2400** (vs record 1.2279 = +0.0121, 1562 steps, 384 ms/step)
  - Skip removal HURTS the record. Asymmetry: gated ME makes the skips cooperative, not redundant.
- B3-9 (heron_3_9.py): no skips + Real Dense — implementation gauntlet:
  - v1 (stack+einsum, fullgraph=True): InductorError OOM. Without skips inductor over-fuses
    dense bmm/einsum with rms_norm → triton kernel Required:168k > Hardware:166k.
  - v2 attempt 1 (scalar dense_layer_scale fusion barrier): still OOM Required:168k.
  - v2 attempt 2 (per-channel barrier): still OOM Required:172k.
  - v3 (Python accumulator loop): InductorError FusedMixOrderReductions assertion (the same
    bug from the earlier session that originally forced the switch to stack+einsum).
  - v4 (`@torch.compiler.disable` free function `_eager_dense_add`, fullgraph=False everywhere):
    COMPILES. But eager-mode dense add → 877 ms/step (2.4× slower than B3-8) → only 685 steps in 600s.
  - **RESULT: 1.4261 final_int6_sliding_window** (severely undertrained, EMA also hurts)
  - **Per-step trajectory tells a different story**: at step 600, B3-9 (1.3507) is ahead of B3-8 (1.3642)
    by **-0.0135**. At step 685, B3-9 (1.3098) is ~equal to B3-8 at step 1000 (1.3105).
    Dense connections give ~46% per-step training-efficiency improvement when not redundant with skips.
- **Verdict**: dense connections DO help per-step in the no-skip regime. The eager-mode implementation
  cost wipes out the win in wallclock-bounded final BPB. A fast (compile-friendly) dense kernel could
  plausibly beat baseline AND B3-8. Reserved as Battle 4 material; out of Battle 3 budget for kernel
  engineering.
- Battle 3 final budget: 10/12. Best win of Battle 3: **B3-8 (1.2323, no skips control), -0.0059 vs baseline**.
- Updated Results.md with full skip-ablation chapter and per-step comparison table.

## Research Battle 2: Multiple Embeddings

### 2026-04-05 - Session Start
- Read all Battle 1 artifacts (Memory, Results, Scratch, Recap)
- Understood architecture: 11 layers, 512 dim, 1024 vocab, tied embeddings, ~27.4M params
- Battle 1 best (heron5.py): 1.2299 BPB (gates: MLP + attn + smear input-dependent)
- Battle 2 baseline = heron5.py
- Tier list: ME > ME smaller dim > ME hashing > gate-boosted winner > Input Projections > Ablation
- Launched ME Tier 1 (me_dim=128, me_num_extra=1): heron_2_1.py
  - Design: secondary embedding (1024x128), per-layer zero-init scalar injection
  - Injection: x += scale_i * pad(me_embed(input_ids), model_dim) at each layer input
  - +131K params (embedding) + 11 scalars
  - RESULT: 1.2292 BPB (final_int6_sw) -- slight improvement over gates-only (1.2299)
  - Size: 13,832,932 bytes (well under 16MB)
  - 1566 steps in 600s, 383ms/step
- Submitted ME dim=64 (heron_2_2.py) and ME hash512 (heron_2_3.py) as batch jobs
- ME dim=64 RESULT: 1.2332 BPB -- WORSE than ME128
- ME hash512 RESULT: 1.2330 BPB -- WORSE than ME128
- Conclusion: full dim (128) and full vocab (1024) needed for ME to work well
- Launched 3 new experiments:
  - B2-4: Gate-boosted ME (heron_2_4.py) - input-dependent gate on ME injection
  - B2-5: Input Projections (heron_2_5.py) - project x0 instead of separate embed
  - B2-6: ME 2 extra embeddings (heron_2_6.py) - 2x128 dim extra embeddings
- Results at step 1000:
  - ME128: 1.3019, ME64: 1.3066, hash512: 1.3073, gated_ME: 1.3006, IP: 1.3153, 2xME: 1.3025
- Final sliding window results:
  - **Gated ME: 1.2279 BPB** -- NEW BEST! Beats admin architecture record (1.2285)!
  - ME128: 1.2292, ME64: 1.2332, hash512: 1.2330, IP: 1.2451 (terrible)
  - 2xME: crashed during sliding window (SLURM timeout)
- Key findings:
  - ME works best at full dim (128) with full vocab (1024)
  - Input-dependent gate on ME injection adds significant value (+0.0013 vs ungated)
  - Input Projections (projecting x0 instead of separate embed) are terrible
  - Hashing the vocab loses too much information
- Launched further experiments:
  - B2-7: Gated ME 2x128 (heron_2_7.py) - gated ME with 2 extra embeddings
  - B2-8: Gated ME dim=192 (heron_2_8.py) - larger embedding dimension
- Final results:
  - Gated ME 2x128: 1.2342 -- WORSE (too slow at 404ms/step, only 1480 steps)
  - Gated ME dim=192: 1.2295 -- close but slightly worse than 128 (fewer steps)
  - **Winner: Gated ME dim=128 (heron_2_4.py) at 1.2279 BPB**
- Updated all documentation: Results.md, recap.md, Memory.md, Scratch.md
- Session complete. 8 experiments run. Best improvement: -0.0103 BPB vs original baseline.
