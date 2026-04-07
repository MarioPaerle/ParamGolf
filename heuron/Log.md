# Heuron Research Group - Log

## Research Battle 4: Is what worked truly transferable? (new heuron baseline, FA3 -> FA2/SDPA)

### 2026-04-07 - Session 4 Start
- Read Heuron.md (TODO: "Is what worked truly transferable?"), Memory.md, Results.md, recap.md, Scratch.md, Log.md.
- Working file: heuron/train_gpt_heuron.py (NEW baseline; previous heron_2_4/heron_3_8 are now in heuron/old/ and renamed `_old` in heuron/).
- The new baseline differs significantly from the old heron series:
  - 2134 lines vs ~1900. Added: ParallelMuon (post-backward reduce-scatter), MTP heads, GPTQ (Hessian-aware int6), `_HessianGPT` non-banked twin model for Hessian collection, autoregressive self-gen calibration, selective +/-1 pruning to TARGET_MB.
  - Default Hyperparameters changes: BIGRAM_VOCAB_SIZE=2048 (was 1536), BIGRAM_DIM=128, **XSA_LAST_N=11 (XSA on ALL layers, not just last 4)**, ROPE_DIMS=16, LN_SCALE=1, VE on layers 9,10 (ve_dim=128), TIE_EMBEDDINGS=1, MUON_MOMENTUM=0.99 + warmup 0.92->0.99 over 1500 steps, GRAD_CLIP_NORM=0.3, late_qat at 0.15.
  - Default off: gated_attention, value_residual, MTP, dtg, lawa.
  - Old battles' "winners" (lightweight gates, gated ME, no-skip) are NOT present in this baseline.
- **CRITICAL fix needed before any run**: this file imports FA3 (`from flash_attn_interface import flash_attn_func as flash_attn_3_func`) at the top. flash_attn_interface (FA3) is not installable on Cineca A100s, so the file cannot even import. Both attention call sites (line 661 in CausalSelfAttention; line 1352 in _HessianAttn) call `flash_attn_3_func(q,k,v,causal=True)`.
- Fix applied (in-place edit, no branch yet — this is an environment fix, not a research win):
  - Removed FA3 import (replaced with two-line comment).
  - In both attention forwards: replaced
    `y = flash_attn_3_func(q, k, v, causal=True)`
    with
    `y = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))` followed by `y = y.transpose(1,2).contiguous()`.
  - The script already calls `enable_flash_sdp(True)` at line ~1596, so SDPA dispatches to FlashAttention-2 on A100. XSA path expects (B,T,H,D) layout — we restore that with `.transpose(1,2).contiguous()` so `_xsa_efficient` works unchanged.
  - `python3 -c "import ast; ast.parse(open('heuron/train_gpt_heuron.py').read())"` passes. `flash_attn_3_func` and the import are gone (only the comment remains).
- Naming convention for this session: `train_gpt_heuron_4_<exp>.py` (Session 4 in the overall research timeline; first session on the heuron-prefix file). The bare-baseline run is on the in-place patched `train_gpt_heuron.py` and will be tracked as B4-0.

### 2026-04-07 - Session 4 - Run B4-0 (bare baseline, FA3-removed)
- Job 39276904 (interactive 30min). Started ~14:07. ~382 ms/step on 4xA100, 1568 steps in 600s wallclock.
- step:1000 val_bpb=1.3061, step:1568 val_bpb=1.2356 (live).
- post-EMA val_bpb=1.2535 (slow EMA, looks worse than live; expected).
- GPTQ Hessian collection from autoregressive self-gen (264s for 64×2048 calibration).
- Selective ±1 prune: unpruned 13.38MB ≤ target 15.9MB → no pruning.
- final_int6_roundtrip val_bpb = 1.2569
- **final_int6_sliding_window val_bpb = 1.23260** (-0.0056 vs old heron baseline 1.2382, +0.0047 vs Battle 2 winner 1.2279)
- Verdict: new baseline (XSA on all 11 layers + MUON warmup + bigram2048 + autoregressive GPTQ) is already much stronger than the old heron baseline, but slightly weaker than heron_2_4 which had Battle 1 gates + Battle 2 gated ME on top of the older base.

### 2026-04-07 - Session 4 - Branches prepared for transferability test
- heuron_4_1.py: B4-0 + Battle 1 lightweight gates (attn_gate_w (num_heads,12), mlp_gate_w (1,12), smear_input_w (1,12), all zero-init -> 2*sigmoid=1.0 no-op). Optimizer: smear_input_w explicitly added; block-level gates picked via CONTROL_TENSOR_NAME_PATTERNS extension `attn_gate_w,mlp_gate_w,mlp_gate,smear_input_w`. AST OK.
- heuron_4_2.py: B4-0 + Battle 2 gated ME (me_dim=128, me_num_extra=1, per-layer scale (zero-init) + per-layer (1,24) gate reading first 12 dims of x and ME signal, init no-op via zero-init). _HessianGPT updated symmetrically so calibration activations match training. Optimizer: me_embeds -> tok_params; me_layer_scales + me_gate_ws -> scalar_params. CONTROL_TENSOR_NAME_PATTERNS extended with `me_layer_scales,me_gate_ws`. AST OK.
- heuron_4_3.py: B4-0 + Battle 1 gates + Battle 2 gated ME (combined). AST OK.
- Submitted B4-1 first (interactive sbatch via salloc). B4-2/B4-3 will follow sequentially as the slot frees.

### 2026-04-07 - Session 4 - Run B4-1 (gates) result
- Job 39283623 (interactive 30min). 390.4 ms/step (1.8% slower than B4-0). 1538 steps in 600s (30 fewer than B4-0).
- model_params:26,994,956 (+1200 from gates: 12*8 + 12 + 12 per layer * 11 layers = 1320 expected, close).
- step:1000 val_bpb=1.3054 (vs B4-0 1.3061, -0.0007).
- step:1538 (live) val_bpb=1.2343 (vs B4-0 step:1568 1.2356, -0.0013).
- post_ema val_bpb=1.2529 (vs B4-0 1.2535, -0.0006). final_int6_roundtrip val_bpb=1.2578 (vs B4-0 1.2569, +0.0009).
- **final_int6_sliding_window val_bpb = 1.23381** (vs B4-0 1.23260, **+0.0012 WORSE**).
- Verdict: gates show a tiny per-step improvement (live BPB) but the 30-step deficit + slightly noisier int6 quantization roundtrip eat the gain on the final metric. NOT TRANSFERABLE on the new baseline.

### 2026-04-07 - Session 4 - Submitted B4-2 / B4-3 in parallel (sbatch)
- Realized I should run multiple jobs in parallel via sbatch instead of sequencing interactively.
- B4-2 (gated ME, heuron_4_2.py): job 39285767 on lrdn0572, RUN_ID=heuron_b4_2_gme. model_params:27,125,103 (+131,347 from 1024×128 ME embedding + per-layer scales + per-layer gates).
- B4-3 (combined gates+ME, heuron_4_3.py): job 39285798 on lrdn0585, RUN_ID=heuron_b4_3_combined. model_params:27,126,303 (+1200 from gates over B4-2, as expected).
- Both started ~16:09 CEST. ~25 min total each (10 min train + ~5 min GPTQ + ~6 min sliding window).
- Inner thought: B4-2 expected target 1.227-1.230 (-0.003 to -0.005 vs B4-0). B4-3 best case 1.226 if gates+ME are independent helpers, worst case ~1.231 if they fight.

### 2026-04-07 - Session 4 - B4-2 / B4-3 results
- **B4-2 (gated ME)**: model_params:27,125,103, 1508 steps, 397.94 ms/step. live val_bpb=1.2359, post_ema=1.2535, int6_RT=1.2573, **final_int6_sw=1.23338** (+0.00078 vs B4-0).
- **B4-3 (combined gates+ME)**: model_params:27,126,303, 1494 steps, 401.68 ms/step. live val_bpb=**1.2331** (best of all four), post_ema=1.2533, int6_RT=1.2584, **final_int6_sw=1.23523** (+0.00263 vs B4-0).
- **Verdict on TODO "Is what worked truly transferable?"**: NO. None of the past Battle 1/2 winners transfer to the new heuron baseline. The bare new baseline (B4-0) is the Battle 4 winner at 1.23260.
- The combined version DOES get the best LIVE val_bpb (1.2331) but loses the wallclock budget battle (74 fewer steps than B4-0) and has slightly worse int6 GPTQ roundtrip damage.
- The new baseline already incorporates much of the value the old gates/ME captured: XSA on ALL 11 layers replaces gated ME's per-layer per-token information; MUON warmup replaces the gate-init smoothing; bigram2048 + VE on 9/10 add token-level signals. So the techniques' "novelty value" was already absorbed by the new baseline.

### 2026-04-07 - Session 4 - Documentation closeout
- Updated Results.md with full Battle 4 chapter (table, findings, implementation details, pseudocode of best run).
- Updated Memory.md with Battle 4 chapter (confirmed principles, untried follow-ups for next agent).
- Updated Scratch.md with environment fix details, sbatch-vs-interactive lesson, ME structure transfer notes, live-vs-final-BPB divergence, untried follow-ups.
- Wrote Recap.md for the next agent.
- Battle 4 budget: 4/4 used. Session complete.

### 2026-04-07 - Session 5 - Battle 4 continuation: XSI variants
- New TODO from user: implement and test 3 XSI (Exclusive Input Attention) variants on the Battle 4 baseline (B4-0 = 1.23260):
  - **XSI v1 (Naive Block)**: per-block, orthogonalize attn_out and mlp_out against their respective block-input. Strongest dose. → file `train_gpt_heuron_5_3.py`.
  - **XSI v2 (Layerwise k=4)**: every 4 layers, orthogonalize residual against snapshot from k layers ago. Mid dose. → file `train_gpt_heuron_5_2.py`.
  - **XSI v3 (Final-logit)**: orthogonalize final residual against `x0` before lm_head. Gentlest. → file `train_gpt_heuron_5_1.py`.
- Branch convention: `train_gpt_heuron_5_<exp>.py`. All 3 mirrored into `_HessianGPT.forward` for GPTQ calibration consistency.
- Compile-time obstacle: v1 and v2 hit triton OOM in fused backward kernel (Required ~170-260KB SMEM, A100 SM has ~167KB). Forward fits, backward doesn't.
- Fix attempt 1 (v2): `.detach()` on the layerwise projection target (BYOL-style stop-grad). → v2 compiles & runs ✓.
- Fix attempt 1 (v3, naive block): `.detach()` alone reduced 170112 → 168096 bytes (still 1.2KB over). Insufficient.
- Fix attempt 2 (v3, naive block): wrapped the orthogonalization in a module-level helper `_xsi_orth(y, ref)` decorated with `@torch._dynamo.disable`. This forces eager execution for the orthogonalization step, breaking the fused backward kernel completely. Slightly slower per-step but avoids the SMEM ceiling.
- 5_1 (XSI v3) result: live val_bpb 1.2403, post_ema 1.2614, int6_RT 1.26503, **final_int6_sw 1.24137** → **+0.00877 vs B4-0**. Clear loss. Same step count and step time, so it's pure per-step quality regression — XSI v3 throws away signal that the tied-embedding lm_head needs.
- Branched 5_7 = learnable-α XSI v3: `xsi_alpha = nn.Parameter(zeros(1))`, scale = `tanh(xsi_alpha)`, init 0 → no-op start. Backup plan if v1/v2 also fail.
- Currently running (4 sbatch parallel + 1 PD):
  - 39293266, 39293529: dud baseline runs (default `train_gpt.py`, the canonical 17M-param OpenAI baseline) — useless for our work, already absorbed compute.
  - 39293653 / 39293683: **5_2 patched ×2** (variance check pair).
  - 39294640: **5_3 with `_xsi_orth` helper** — pending in queue.
- Lesson logged: when torch.compile fuses too much around an eager-natural op, prefer `@torch._dynamo.disable` over manual `.detach()` plumbing. The detach trick saved ~1% of the SMEM budget; the disable decorator solves the problem categorically.

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

### 2026-04-07 - Session 6 Start (Battle 4 Continue)
- Read Recap/Memory/Results/Scratch. Current winner: 5_7 (XSI v3 + learnable scalar α) at 1.23188 BPB.
- 5_8 (per-channel α): 1.23423 worse. 5_9 (XSI v2 + α): 1.24706 worse. 5_10 (per-block α): 1.23413 worse.
- Picked from Recap "what to try next": 5_11 = 5_7 with projection target `F.normalize(self.final_norm(x0))` instead of `F.normalize(x0)`. Fixes magnitude mismatch between RMS-normalized x and unnormalized x0.
- Created heuron/train_gpt_heuron_5_11.py (copy of 5_7, 3 lines changed: GPT.forward, GPT.forward_logits, _HessianGPT.forward). Mirror is critical so GPTQ calibration matches training.
- Submitted via sbatch: job 39307873, RUN_ID=heuron_5_11_xsi_finalnorm_target. Pending in priority queue.
- Inner thought: expect step_avg ~383ms (same as 5_7, no extra work — final_norm is already computed once per forward, this just reuses it). final_int6_sw expected in [1.2305, 1.2335] range. The change is small enough that it might be pure noise; if it ties or beats 5_7 we keep it, if it regresses we know x0 raw was the right target.

### 2026-04-07 - Session 6 Continued (Battle 4 — beating 5_7)
- 5_11 (final_norm(x0) projection target): 1.23154 — marginal win over 5_7 (1.23188).
- 5_12 (XSI applied AFTER final_norm): 1.23312 ❌
- 5_13 (5_7 + full Battle 1 gates ported): 1.23239 ❌ (old gates don't transfer)
- 5_14 (NOV last 4 layers): 3 attempts, all crashed with InductorError SMEM 221k > 167k. Avoiding repeat_interleave + dynamo disable + expand+reshape all failed. NOV needs a different strategy.
- 5_15 (cross-sublayer MLP gate: attn_out[..,:12] gates the MLP contribution): **1.22817 — biggest single win, −0.00371 vs 5_7.** The unlock was reading from *fresh sub-layer outputs* instead of the static residual.
- 5_16 (gated bigram): 1.23376 ❌
- 5_17 (5_11 + per-channel α D=512): 1.23440 ❌ (per-channel α confirmed bad twice now)
- 5_18 (mid + final XSI): 1.24033 ❌ (extra XSI insertion regresses badly)
- 5_19 (5_15 + 5_11 final_norm target combined): 1.22871 — ties 5_15
- 5_20 (5_15 with x_mlp_gate_w widened to 24 dims, cat input): 1.22830 — ties 5_15
- 5_21 (5_15 + per-head Battle 1 attn output gate, 8×12 params): **1.22803 — NEW BEST, −0.00385 vs 5_7.** Confirms that the cross-sublayer MLP gate stacks with a per-head attention output gate.
- 5_22 (5_15 + Battle 1 smear_input_w gate): in flight
- 5_23 (5_21 + classic Battle 1 mlp_gate from x_out, multiplicative on top of cross-sublayer): in flight
- 5_24 (5_21 with attn_gate_w widened to 24 input dims): in flight
- Updated Results.md with 5_11..5_20 outcomes and the 5_15 KEY FINDING. Recap.md / Memory.md / Scratch.md updated with the cross-sublayer routing principle (read from fresh sub-layer outputs, not static residual).

### 2026-04-07 - Session 6 cont'd (scaling the per-head attn gate)
- 5_22 (5_15 + smear_input_w): 1.23439 ❌
- 5_23 (5_21 + classic mlp_gate stacked): 1.23417 ❌
- 5_24 (5_21 with attn_gate_w widened to 24 dims): 1.22782 ✅ -0.00021 vs 5_21
- 5_25 (attn_gate_w widened to 48 dims): **1.22661 ✅✅ NEW BEST, -0.00121 vs 5_24, -0.00527 vs 5_7**
- 5_26 (5_24 + wide cross-mlp gate 24 dims): 1.23186 ❌ (MLP gate widening hurts; only attn-gate widening pays)
- 5_27 (5_24 + per-head qgain modulation gate): CRASH (state_dict mismatch on quant reload)
- KEY PATTERN: scaling the per-head attn output gate's *input width* monotonically improves results.
  Submitted scaling sweep 5_28 (64), 5_29 (96), 5_30 (128) to find the saturation point.
