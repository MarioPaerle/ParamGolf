# XSI (eXclusive Self-Input) Variants — Pseudocode and Lessons

XSI is the dual of XSA. Where **XSA** (in baseline) orthogonalizes the *attention output* against
the per-head **value** direction, **XSI** orthogonalizes the *residual stream* against the per-token
**input embedding** direction `x0`. The intuition: x0 is what the lm_head already "knows" about the
token, so subtracting its component from `x` could free the residual to carry purely *contextual*
information.

This file walks through every XSI variant we tested (5_1 → 5_18) with pseudocode and what we
learned. Current Battle 4 record (post-XSI work) is **5_25 = 1.22661 BPB**, which still keeps the
5_7 XSI v3 + learnable α component at the head of the network.

---

## Notation
- `x`: residual stream, shape `(B, T, D)`
- `x0`: input embedding (smeared, post-`smear_input`), shape `(B, T, D)`
- `final_norm`: RMSNorm at the end of the model
- All gates `α` are zero-initialized; `tanh(α=0) = 0` so XSI starts as a no-op (identity at step 0).
- **`*` is always pointwise (Hadamard) multiplication** with PyTorch broadcasting semantics.
  The dot product along the last dim is written explicitly as `(a * b).sum(dim=-1, keepdim=True)`.
  So the canonical XSI projection-removal `x - <x, x0_n> * x0_n` reads as
  `x - (x * x0_n).sum(dim=-1, keepdim=True) * x0_n`, where the inner `*` is Hadamard before the
  reduction and the outer `*` is broadcast Hadamard of the `(B,T,1)` scalar coefficient against
  the `(B,T,D)` direction `x0_n`.

---

## 5_1: Rigid XSI v3 — final-layer orthogonalization, **α=1** (fixed)

```python
# applied right before final_norm + lm_head, in GPT.forward and GPT.forward_logits
x0_n = F.normalize(x0, dim=-1)
x = x - (x * x0_n).sum(dim=-1, keepdim=True) * x0_n   # rigid projection removal
x = self.final_norm(x)
```

**Result**: 1.24137 BPB ❌ (+0.00877 vs B4-0 baseline)

**Why it failed**: with tied embeddings, the lm_head logits are roughly aligned with each token's
embedding direction. Removing the entire `x0` component throws away the model's "starting from
this token" signal that the head needs for next-token prediction. XSA doesn't suffer this because
its projection target (the value `v`) is *learned* and *layer-specific* — the model can route
around it. `x0` is fixed and per-token; the model can't recover from rigid orthogonalization
against a fixed reference.

**Lesson**: rigid orthogonalization against fixed reference directions is harmful at the head.

---

## 5_2: XSI v2 — layerwise k=4 (rigid)

```python
# in GPT.forward, every 4 layers snapshot the residual stream
saved_inputs = []
for i, block in enumerate(self.blocks):
    if i % 4 == 0:
        saved_inputs.append(x.detach())  # detach prevents BYOL-style giant backward kernel
    if i % 4 == 3 and len(saved_inputs) >= 2:
        snap = saved_inputs[-2]  # snapshot from 4 layers ago
        snap_n = F.normalize(snap, dim=-1)
        x = x - (x * snap_n).sum(dim=-1, keepdim=True) * snap_n
    x, _ = block(x, x0, ...)
```

**Result**: 1.24715 BPB ❌ (+0.01455, also slower at 403 ms/step)

**Why it failed**: same rigidity problem at every k-th layer, *plus* the BYOL stop-grad cost. The
fix to use `.detach()` was necessary to avoid `InductorError: SMEM 260k > 167k` in the backward
pass — without it, torch.compile fuses the cross-layer gradient flow into one giant kernel.

**Lesson**: cross-layer projection targets need detach() to keep `torch.compile` happy. Even with
the detach fix, repeated rigid orthogonalization just compounds the damage.

---

## 5_7: XSI v3 + **single learnable α** (the original Battle 4 winner) ⭐

```python
# in GPT.__init__:
self.xsi_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # single scalar, init 0

# applied right before final_norm + lm_head:
x0_n = F.normalize(x0, dim=-1)
xsi_a = torch.tanh(self.xsi_alpha).to(dtype=x.dtype)   # bounded in (-1, 1), starts at 0
x = x - xsi_a * (x * x0_n).sum(dim=-1, keepdim=True) * x0_n
x = self.final_norm(x)
```

**Result**: **1.23188 BPB ✅** (−0.00072 vs B4-0 baseline 1.23260)

**Why it worked**: starts as a no-op (α=0 → identity). The model can dial up XSI strength only if
useful, and stops short of full removal. Same architecture as 5_1 but **+1 scalar parameter** turns
a +0.00877 regression into a −0.00072 win. Net swing from 5_1 → 5_7 = **0.00949 BPB**.

**General principle worth memorizing**: when adding a destructive op (subtraction, mask, dropout),
gate it behind a `tanh`/`sigmoid`-bounded learnable scalar, init at zero. Cost: 1 param. Reward:
the model can opt in only if helpful, and the option to recover the baseline at α=0 is preserved.

This single scalar is one of the most cost-effective wins of the entire research program.

---

## 5_8: XSI v3 + **per-channel α** (D=512 params)

```python
self.xsi_alpha = nn.Parameter(torch.zeros(D, dtype=torch.float32))  # per-channel, init 0

# applied identically to 5_7 but α broadcasts per-channel
xsi_a = torch.tanh(self.xsi_alpha).to(dtype=x.dtype)   # shape (D,)
proj = (x * x0_n).sum(dim=-1, keepdim=True) * x0_n
x = x - xsi_a[None, None, :] * proj
```

**Result**: 1.23423 BPB ❌ (+0.00235 vs 5_7)

**Why it failed**: per-channel selectivity over-fits the early-step gradient noise; with only ~1500
steps and ~26M params, +511 extra trainable params for a destructive op aren't trained well enough
to be net positive. The single scalar finds the global sweet spot more reliably.

**Re-confirmed**: 5_17 (per-channel α applied on top of 5_11's better projection target) also lost
at 1.23440. **Don't try per-channel α a third time.**

---

## 5_9: XSI v2 layerwise + **per-position learnable α**

```python
# 11 per-block scalars + final scalar, all init 0, all tanh-bounded
for i, block in enumerate(self.blocks):
    x, _ = block(...)
    if i % 4 == 3:
        snap = saved_inputs[-2]
        snap_n = F.normalize(snap, dim=-1)
        a = torch.tanh(self.xsi_alphas[i])
        x = x - a * (x * snap_n).sum(dim=-1, keepdim=True) * snap_n
```

**Result**: 1.24706 BPB ❌

**Why it failed**: even with learnable α, the layerwise variant was already structurally
bad (5_2 was −0.014). The α gives the model the option to disable each block's XSI individually,
but the repeated operation still costs steps (slower kernel) and the snapshots still inject
training noise. Conclusion: XSI v2 is not rescuable.

---

## 5_10: XSI v3 + **per-block learnable α stack**

```python
# in GPT.__init__:
self.xsi_block_alphas = nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
self.xsi_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # original final scalar

# in GPT.forward, after every block:
for i, block in enumerate(self.blocks):
    x, _ = block(x, x0, ...)
    a = torch.tanh(self.xsi_block_alphas[i]).to(dtype=x.dtype)
    x = x - a * (x * x0_n).sum(dim=-1, keepdim=True) * x0_n

# final-layer XSI still applied (5_7 style)
xsi_a = torch.tanh(self.xsi_alpha).to(dtype=x.dtype)
x = x - xsi_a * (x * x0_n).sum(dim=-1, keepdim=True) * x0_n
```

**Result**: 1.23413 BPB ❌

**Why it failed**: 11 extra XSI insertion points means 11 extra opportunities to bleed signal even
when each individual α stays small. The final-layer 5_7 pattern is special because it acts ONCE
on the output state, just before the lm_head. Inserting XSI in every block is a different
operation entirely — the residual stream needs `x0`-aligned components flowing through it to
build later features.

**Lesson**: the final-layer position of XSI is load-bearing. It's a *post-processing* operation,
not an in-block operation.

---

## 5_11: 5_7 + **`final_norm(x0)` projection target** (instead of raw `x0`)

```python
# only the projection target changes
x0_n = F.normalize(self.final_norm(x0), dim=-1)   # was: F.normalize(x0, dim=-1)
xsi_a = torch.tanh(self.xsi_alpha).to(dtype=x.dtype)
x = x - xsi_a * (x * x0_n).sum(dim=-1, keepdim=True) * x0_n
x = self.final_norm(x)
```

**Result**: 1.23154 BPB ✅ (−0.00034 vs 5_7) — marginal but real improvement

**Why it worked**: the residual stream `x` has been RMS-normalized by `final_norm` shortly after,
but the projection happens BEFORE that norm. Using `final_norm(x0)` as the target makes the
projection direction live in the same scaled space the residual will be in, fixing a small
magnitude mismatch. The win is small (within noise) but consistent.

**Caveat**: when stacked with 5_15 (giving 5_19), the combined improvement saturates — the
projection-target tweak and the cross-sublayer gate are pulling on overlapping signals.

---

## 5_12: XSI applied **AFTER** `final_norm` (order swap)

```python
x = self.final_norm(x)     # final_norm FIRST
x0_n = F.normalize(self.final_norm(x0), dim=-1)
xsi_a = torch.tanh(self.xsi_alpha).to(dtype=x.dtype)
x = x - xsi_a * (x * x0_n).sum(dim=-1, keepdim=True) * x0_n   # XSI AFTER
# (no second final_norm)
```

**Result**: 1.23312 BPB ❌ (+0.00124 vs 5_7)

**Why it failed**: applying XSI to the already-normalized residual is mathematically redundant
with the original 5_7 placement (which then renormalizes), but it changes the gradient flow:
the lm_head now sees a residual stream that has been *post-norm-perturbed*, breaking RMS
invariants the lm_head implicitly relies on for stable gradients.

**Lesson**: order matters. XSI must be **before** `final_norm`, not after.

---

## 5_18: 5_11 + **second XSI applied at encoder→decoder midpoint**

```python
# in GPT.__init__:
self.xsi_alpha_mid = nn.Parameter(torch.zeros(1, dtype=torch.float32))
# (xsi_alpha for the final position is unchanged)

# in GPT.forward, between encoder and decoder layers:
for i in range(self.num_encoder_layers):
    x, _ = self.blocks[i](x, x0, ...)

# 5_18: mid-network XSI between encoder and decoder
x0_n_mid = F.normalize(self.final_norm(x0), dim=-1)
xsi_a_mid = torch.tanh(self.xsi_alpha_mid).to(dtype=x.dtype)
x = x - xsi_a_mid * (x * x0_n_mid).sum(dim=-1, keepdim=True) * x0_n_mid

for i in range(self.num_decoder_layers):
    x, _ = self.blocks[self.num_encoder_layers + i](x, x0, ...)

# original 5_11 final XSI still applied
```

**Result**: 1.24033 BPB ❌ (+0.00845 vs 5_7) — bad regression

**Why it failed**: same mistake as 5_10 in spirit. Adding extra XSI insertion points within the
network (rather than only at the head) interferes with the internal flow of `x0`-aligned
information that later layers depend on. Even with a learnable α, the model can't fully suppress
the damage because the bias of α=0 isn't reached at convergence.

**Lesson confirmed twice now (5_10, 5_18)**: XSI is a **head-only** operation. Don't insert it
inside the network.

---

## Where does the current Battle 4 record (1.22661) sit?

The current record is **5_25 = 1.22661 BPB**, but **5_25 is not an XSI variant** — it's a
per-head attn-output-gate widening (W=48) stacked on the 5_15 cross-sublayer MLP gate. Both 5_15
and 5_25 keep the 5_7 XSI v3 + scalar α component intact at the head of the network. So the
XSI mechanism contributes one fixed building block to every record-class run since 5_7; the
*new* progress comes from gating, not from changing XSI.

Stack composition of the current record (5_25):

```text
GPT.forward(input_ids):
    x  = tok_emb + bigram + ...
    x0 = smear_input(x)              # input embedding, frozen post-smear

    for block in blocks:
        x = block(x, x0)             # contains:
                                     #   - per-head attn output gate W=48 (5_25, NEW)
                                     #   - cross-sublayer MLP gate from attn_out (5_15)
                                     #   - XSA on all 11 layers (baseline)
                                     #   - resid_mix, attn_scale, mlp_scale (baseline)

    # 5_7 XSI v3 + learnable scalar α (still here, unchanged):
    x0_n = F.normalize(x0, dim=-1)
    xsi_a = torch.tanh(self.xsi_alpha)
    x = x - xsi_a * (x * x0_n).sum(dim=-1, keepdim=True) * x0_n
    x = self.final_norm(x)

    return F.linear(x, tok_emb.weight)   # tied
```

So when you read the XSI table below, the BPB column is the score of the **full model** with that
particular XSI mechanism plugged in — but with the cross-sublayer gating wins NOT yet present
(those came in 5_15+). The Δ vs 5_7 column compares only XSI-mechanism choices on a fixed
otherwise-baseline architecture. To see the *total* stacked progress (XSI + gating), see
[Results.md](Results.md) Battle 4 R2/R3 tables.

For reference, the absolute leaderboard across all of Battle 4 R2/R3:

| Branch | BPB | Mechanism delta from previous |
|--------|-----|-------------------------------|
| B4-0 (bare new heuron baseline) | 1.23260 | — |
| **5_7** XSI v3 + learnable α | **1.23188** | +XSI scalar gate at head (this file) |
| 5_11 5_7 with `final_norm(x0)` projection target | 1.23154 | tweak XSI projection target |
| 5_15 5_7 + cross-sublayer MLP gate from `attn_out[..,:12]` | 1.22817 | +gating, not XSI |
| 5_21 5_15 + per-head attn output gate W=12 | 1.22803 | +gating, not XSI |
| 5_24 5_21 with attn_gate_w widened to W=24 | 1.22782 | scaling, not XSI |
| **5_25** **5_24 with attn_gate_w widened to W=48** | **1.22661** | scaling, not XSI |
| 5_28 (W=64), 5_29 (W=96), 5_30 (W=128) | in flight | scaling sweep |

The XSI contribution to the record is **−0.00072** (5_7 over B4-0). Everything below 1.23188 came
from the cross-sublayer gating work, not from changing XSI.

---

## Summary of XSI variants

| # | Variant | α | Position | BPB | Δ vs 5_7 (1.23188) | Verdict |
|---|---------|---|----------|-----|---------------------|---------|
| 5_1 | v3 final | 1 (rigid) | head | 1.24137 | +0.00949 | ❌ rigid kills it |
| 5_2 | v2 layerwise k=4 | 1 (rigid) | every 4 layers | 1.24715 | +0.01527 | ❌ |
| **5_7** | **v3 final** | **scalar learn** | **head** | **1.23188** | **0** | ⭐ **the Battle 4 R2 winner** |
| 5_8 | v3 final | per-channel D=512 | head | 1.23423 | +0.00235 | ❌ over-parameterized |
| 5_9 | v2 layerwise + α | per-block | every 4 layers | 1.24706 | +0.01518 | ❌ unrescuable |
| 5_10 | v3 + per-block α stack | 11+1 scalars | every block + head | 1.23413 | +0.00225 | ❌ in-block XSI hurts |
| 5_11 | 5_7 with `final_norm(x0)` target | scalar | head | **1.23154** | **−0.00034** | ✅ small win |
| 5_12 | XSI after final_norm | scalar | head (post-norm) | 1.23312 | +0.00124 | ❌ wrong order |
| 5_17 | 5_11 + per-channel α | D=512 | head | 1.23440 | +0.00252 | ❌ confirms 5_8 |
| 5_18 | 5_11 + mid-network 2nd XSI | 2 scalars | mid + head | 1.24033 | +0.00845 | ❌ confirms 5_10 |

---

## Distilled lessons

1. **Rigid orthogonalization is harmful at the head**, because it removes the tied-embedding signal
   the lm_head needs for next-token prediction. Always gate XSI behind a learnable bounded scalar
   init at zero — this is the single most important XSI lesson and generalizes to any destructive
   op.

2. **XSI is a head-only operation.** In-block insertion (5_10, 5_18) and layerwise insertion (5_2,
   5_9) all regress, even when the per-position α is learnable. The residual stream needs
   `x0`-aligned components flowing through it to build later features.

3. **Use `final_norm(x0)` as the projection target**, not raw `x0` (5_11 over 5_7 by 0.00034).
   Aligns the projection direction with the post-norm space the residual will be in.

4. **XSI must be applied BEFORE `final_norm`**, not after (5_12 vs 5_7).

5. **Per-channel α is dead** — tested twice (5_8, 5_17), regresses both times. The single scalar
   finds the global sweet spot more reliably with limited training steps.

6. **The single scalar XSI (5_7 / 5_11) is still in the current record stack** (5_25 = 1.22661).
   It contributes a small but stable improvement, and combines additively with the cross-sublayer
   gating wins (5_15, 5_21, 5_24, 5_25) that drove the rest of the Battle 4 R3 progress.

7. **General destructive-op gating principle** (worth committing to a CLAUDE-style memory): any
   time you'd add a subtraction, mask, dropout, or other lossy op, gate it behind
   `tanh(self.something_alpha)` with `something_alpha = nn.Parameter(torch.zeros(1))`. Cost: 1
   parameter. Benefit: the model can opt in only if helpful, and you preserve the option to
   recover the baseline at α=0 with no surgery.
