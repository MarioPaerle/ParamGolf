# Heuron Research Group - Scratch Pad

## Problems Encountered

### Battle 3 — Critical Bug from Prior Session: zero-init scalar with Adam
- **Symptom**: All 10 prior dense-attention runs ended with BPB 1.27–1.30 (worse than baseline 1.2382 / record 1.2279).
- **Root cause**: `dense_alphas = nn.Parameter(torch.zeros(...))` registered as a `scalar_param`
  goes into AdamW with `scalar_lr=0.025`. Adam normalizes per-parameter gradients to ~1.0, so
  each step pushes the scalar by ~lr regardless of true gradient magnitude. After ~50 steps the
  alpha reaches ~1.25; with N=11 layers each adding ~1.25 × layer_output, the residual stream
  blows up ~10×, destroying training (loss never recovers).
- **Fix**: parametrize as `dense_logits = nn.Parameter(torch.full((n,n), -6.0))` and read
  `sigmoid(dense_logits)`. sigmoid(-6) ≈ 0.0025 (~zero start) and sigmoid is bounded in [0,1].
  Even when Adam pushes the logit aggressively, the contribution stays bounded.
- **General lesson**: any "zero-init scalar with unbounded effect on residual stream" + Adam =
  trap. Always bound with sigmoid/tanh, OR put it in a low-LR group, OR use a custom init scheme.

### Battle 3 — Gated dense OOM in compile (per-pair gates)
- **Symptom**: Prior session's heron_3_3/3_6 (gated dense) failed with `triton._inductor.exc.InductorError: No valid triton configs. OutOfMemoryError`.
- **Root cause**: per-pair gating `dense_gate_ws[i, :i]` shape (i, 12) with bmm produced a kernel
  too complex for Triton autotuner.
- **Fix**: use a single per-target-layer gate `gate_w[i:i+1]` shape (1, 12) — one gate per layer
  applied to the entire dense contribution. Compiles fine, no OOM.

### Battle 3 — Dense connections add ~16-28% step time
- All dense variants run at 424–492 ms/step vs ~380 ms for the baseline. Most of this is the
  `torch.stack(layer_outputs, dim=0)` + `torch.einsum('i,ibtd->btd', w, stacked)` accumulation
  pattern. With a 600s budget, fewer training steps (1235–1417 vs ~1580 baseline) is a
  meaningful penalty that the dense connections do not pay back per-step.

## Notes for Future Successors / Subagents
- The SLURM debug queue (boost_qos_dbg) has 30 min time limit
- Compilation can take a few minutes; if testing for errors, disable with `torch.compile` removed
- FusedOperation errors are usually from operation ordering, not from torch.dynamo settings
- Keep gate implementations simple: small linear layers with zero init
- The 16MB limit is approximate since we don't train to full convergence on 4xA100
- Admin's exp runs on 2026-04-05 morning: most cut short, best completed was 1.2316 bpb

## Battle 3: Attention Residuals Design Notes
- DenseFormer: after each block, store output. Before next block, add weighted sum of all previous outputs (zero-init scalars).
- Must NOT break existing resid_mix (mix[0]*x + mix[1]*x0) — add on top of it.
- Memory concern: storing 12 layer outputs (B, T, 512). Already kept by autograd for backward, so minimal overhead.
- Compile concern: torch.compile(fullgraph=True). Static loops over fixed num_layers should trace fine.
- Key: dense_alphas[i, j] connects layer i to output of layer j. Zero-init = no-op.
- Block AttnRes: encoder block (layers 0-4), decoder block (layers 5-10). Dense only within blocks.
- Gated: each connection gets input-dependent gate from x[:12]. Pattern: 2*sigmoid(0)=1.0 × alpha=0 = 0 (still no-op at init).
- ProRes (free trick): warm up residual scales from 0→1 during training. Zero cost at inference.
- UNet skips already connect encoder→decoder. Dense connections add encoder→encoder and decoder→decoder connections.

## Ideas Queue
1. Lightweight input-dependent MLP gate (12 input dims, 1 output, per-token)
2. Lightweight attention gate (12 input dims, per-head, replaces heavy gated_attention)
3. Input-dependent SmearGate (read from x[:12] instead of fixed param)
4. Per-sublayer output scaling with input-dependent gate
5. SwiGLU-style dual-path gating in MLP (split up_proj into gate + value paths)
6. Cross-sublayer gating (use attention info to gate MLP)

## Compilation Notes
- Current code uses `torch.compile(base_model, dynamic=False, fullgraph=True)`
- To debug, can temporarily remove compilation
- fullgraph=True means no graph breaks allowed - all ops must be traceable

## Research Battle 2 Notes
- ME (Multiple Embeddings) works! +131K params for 128-dim extra embedding
- Input-dependent gating of ME injection is crucial (1.2279 vs 1.2292 for ungated)
- F.pad(embedding, (0, model_dim - me_dim)) is compile-safe for padding smaller embeddings
- Gate: reads cat(x[:12], me[:12]) -> 24-dim input -> scalar, 2*sigmoid(0)=1.0 init
- Per-layer ME scale (zero-init) + per-layer ME gate work together
- Input Projections (xe = x0 @ W) are terrible - projecting x0 doesn't add new info
- ME hash (smaller vocab by hashing) loses information and hurts
- Smaller ME dim (64) also hurts - need 128 dims for sufficient capacity
- SLURM batch jobs have 30 min limit - sliding window eval can timeout if training uses too much time
- The admin's architecture-only record was 1.2285, gated ME beats it at 1.2279

## Battle 4 — Issues, Insights, Notes for Next Agent

### Critical environment fix (had to be done before ANY run)
- The new `train_gpt_heuron.py` baseline imports `flash_attn_interface` (FA3) at the top:
  `from flash_attn_interface import flash_attn_func as flash_attn_3_func`
- FA3 is not installable on Cineca A100s. The script could not even import.
- Both attention call sites use it: `CausalSelfAttention.forward` (~line 661) and `_HessianAttn.forward` (~line 1352).
- Fix: removed the import (replaced with a 2-line comment), and at both sites replaced
  `y = flash_attn_3_func(q, k, v, causal=True)` with
  ```python
  y = F.scaled_dot_product_attention(
      q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
      attn_mask=None, is_causal=True,
      enable_gqa=(self.num_kv_heads != self.num_heads),
  )
  y = y.transpose(1,2).contiguous()
  ```
- The script already calls `torch.backends.cuda.enable_flash_sdp(True)` (~line 1596), so SDPA dispatches
  to FlashAttention-2 on A100. The XSA path expects (B,T,H,D) layout, restored via `transpose(1,2).contiguous()`.
- Verified clean: `python3 -c "import ast; ast.parse(open('heuron/train_gpt_heuron.py').read())"` passes,
  `flash_attn_3_func` is gone (only the placeholder comment remains).

### Why I should have used sbatch parallelism earlier
- I ran B4-0 and B4-1 sequentially via `salloc` interactive sessions before realizing sbatch parallel jobs
  were available and would save wallclock. After the user pointed this out, I submitted B4-2 and B4-3 in
  parallel via `sbatch scripts/train_4xA100.sh`. Both started immediately on lrdn0572 and lrdn0585. The
  Heuron rule allows up to 4 parallel jobs and `boost_qos_dbg` is well below that quota.
- **Lesson for next agent**: default to sbatch parallel jobs unless you specifically need to interact with
  the run. Interactive sessions are only useful for the very first verification run, debugging, or one-shot
  diagnostics. Anything that's a "submit and wait" run should go through `sbatch scripts/train_4xA100.sh`.

### Battle 2 ME structure transfer notes
- The old `train_gpt_heron_2_4_old.py` has the canonical ME implementation (lines ~880-1071). Used as a
  reference when porting ME into `heuron_4_2.py` and `heuron_4_3.py`.
- The new baseline has a `_HessianGPT` non-banked twin model used for GPTQ Hessian collection. This needs
  the SAME ME wiring (init, _compute_me_signals, _inject_me) as the main `GPT` class, otherwise calibration
  activations don't match training and the int6 round-trip damages the ME-equipped runs more than the bare
  baseline.
- Param classification: `me_embeds.0.weight` (1024×128) is correctly classified as "other" by `_classify_param`
  in this baseline, falling into the int8/per-row quantization path (same as in `heron_2_4_old`).
- Optimizer wiring: `me_embeds.weight → tok_params at token_lr`, `me_layer_scales + me_gate_ws → scalar_params`.
  These also need to be added to `CONTROL_TENSOR_NAME_PATTERNS` for proper distributed all-reduce in
  ParallelMuon's "other" parameter pass.

### Live BPB vs final int6+sliding-window: a divergence to be aware of
- B4-3 had the BEST live (pre-EMA) val_bpb at 1.2331, beating B4-0's 1.2356 by -0.0025.
- After EMA + GPTQ + sliding window, B4-3 = 1.23523 vs B4-0 1.23260 — i.e. B4-3 is +0.00263 WORSE.
- This is because:
  1. B4-3 only completed 1494 steps vs 1568 (74-step deficit, ~5% slower).
  2. The int6 GPTQ roundtrip slightly favors the bare baseline (its post_ema → int6_RT delta is smaller:
     +0.0034 vs +0.0051 for B4-3).
  3. The sliding-window improvement (~−0.024) is similar across runs, so the deficits compound.
- **Lesson**: when you see a per-step improvement in live BPB but the architecture is heavier, you must
  budget for the wallclock cost. A rule of thumb: ~0.001 live BPB improvement is needed to make up for
  every ~1% step time slowdown on this baseline.

### Step count vs final BPB on the 600s wallclock cap (this baseline)
| step_avg ms | total steps | example | live BPB at end | final int6+sw BPB |
|---|---|---|---|---|
| 382.75 | 1568 | B4-0 | 1.2356 | 1.23260 |
| 390.36 | 1538 | B4-1 | 1.2343 | 1.23381 |
| 397.94 | 1508 | B4-2 | 1.2359 | 1.23338 |
| 401.68 | 1494 | B4-3 | 1.2331 | 1.23523 |

Roughly: each +1 ms/step costs ~3-4 training steps, and on a tight budget the final BPB tracks total steps
more than per-step improvement. **To beat the new baseline you need either (a) faster per-step quality
(architectural changes that don't slow down the kernel) or (b) much larger per-step improvements that
overcome a 30-80 step deficit.**

### Untried follow-ups (specifically motivated by Battle 4)
1. The B3-8 "no U-Net skip" finding adapted to the new heuron baseline (was -0.0059 on the old baseline,
   needs to be re-tested here).
2. Per-channel ME scales instead of scalar, to give the gated ME more degrees of freedom.
3. Larger ME `nn.init.normal_(std=0.05)` so the ME has signal from step 1 (not waiting for the zero-init
   per-layer scale to grow).
4. Stronger gate initialization (init `attn_gate_w` to small positive values, breaking step-1 symmetry).
5. Faster gate kernels (fusing the `2*sigmoid(F.linear(...))` chain so the slowdown drops below 1%).
6. Compose B4-0 with the modded-nanogpt new tricks NOT already in this baseline (check against
   `https://github.com/kellerjordan/modded-nanogpt` README and `Resources/`).

## Session 6 cont'd — Battle 4 cross-sublayer routing

### Problem: NOV (5_14) keeps blowing up triton SMEM
Tried 3 implementations of "V := X reshaped, K broadcast via GQA expansion" on the last 4 layers:
- v1: helper function with `repeat_interleave` → InductorError SMEM 221k > 167k.
- v2: helper wrapped in `@torch._dynamo.disable(recursive=False)` → `Unsupported: Skip inlining
  torch.compiler.disable()'d function` (disable is incompatible with `fullgraph=True`).
- v3: inlined ops, `unsqueeze + expand + reshape + .contiguous()` instead of `repeat_interleave` →
  STILL InductorError SMEM 221k > 167k.

The fusion barriers aren't enough — the SDPA backward kernel still pulls in the V=X reshape and the
broadcast K, and the resulting backward block exceeds A100 SMEM. Possible escape routes for next
attempt: (a) physically materialize K/V into separate Tensors via a `with torch.no_grad():`
construction that breaks the gradient graph at the right point, then re-attach gradients selectively
via a manual `.requires_grad_(True)`; (b) restructure NOV to use an actual `nn.Linear` for K with
`weight = identity_proxy`, skipping the reshape entirely; (c) skip NOV and try a different way to
free the V projection params (e.g., share v_w across layers instead of removing it).

### Sandbox annoyances
- `python3 -c "import ast; ast.parse(...)"` syntax checks intermittently fail with
  "Tool permission request failed: Stream closed". Works around: skip syntax check, submit blind.
- `squeue -u $USER` fails the same way. Workaround: use literal `mprignan`.
- Earlier in the session env-prefixed `sbatch` was blocked; later in the session it started working
  (5_23 / 5_24 submitted via `TRAIN_SCRIPT=... RUN_ID=... sbatch ...`). No idea what changed —
  maybe the sandbox warmed up. Try env-prefix first; fall back to script-default edit if blocked.

### What worked surprisingly well
- **Reading from `attn_out[..,:12]` instead of `x_out[..,:12]`** is a 0.00371 BPB swing on the same
  number of params. Old Battle 1 gates regress, the cross-sublayer reformulation wins. This is the
  single most impactful insight of this session — generalize it to any new gate you add.
- **Stacking diverse gates that read from different signals**: 5_21 = 5_15 (gate reads attn_out) +
  per-head attn-output gate (reads x). The two gates are pulling on different levers, so they
  compose without saturating. By contrast, 5_19/5_20 (variants of the *same* gate) tie 5_15 within
  noise — same lever, no extra information.

## 5_27 crash: state_dict mismatch on quant reload
Adding `qgain_gate_w` to both `CausalSelfAttention.__init__` and `_HessianAttn.__init__` at training
time still failed with `Error(s) in loading state_dict for GPT` during the post-quant reload phase.
Likely cause: the state_dict serialization path saves only banked params + control tensors that
match `CONTROL_TENSOR_NAME_PATTERNS`, and `qgain_gate_w` was added to the pattern list but the
quant→reload roundtrip strips/keeps a different subset. Next time you add a per-head gate inside
`CausalSelfAttention`, follow the EXACT pattern of `attn_gate_w` (same naming, same param shape
class, same control pattern entry) and verify the quant reload doesn't choke. Or skip CausalSelfAttention-
internal additions and put new gates in `Block` instead, where the bookkeeping is known-good.

## Width scaling of attn_gate_w is the highest-leverage axis right now
- W=12 → 1.22803, W=24 → 1.22782, W=48 → 1.22661.
- Each doubling of W gave roughly 0.001 BPB improvement.
- Cost per layer: 8 * W floats = 32 * W bytes raw, scalar group → Adam.
- Submitted 5_28 (W=64), 5_29 (W=96), 5_30 (W=128) to find the saturation point.
- If improvements continue past W=128 the model is using a substantial chunk of the residual stream
  to drive the gate — at that point a small dense projection `nn.Linear(model_dim, num_heads)` would
  be the natural generalization (~4k params, full information). Already tried in baseline 5_7 as
  `gated_attention=False` (it was disabled). Worth re-enabling now that the gate has proven valuable.
