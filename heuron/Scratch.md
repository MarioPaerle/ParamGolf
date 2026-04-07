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
