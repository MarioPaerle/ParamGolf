# Attention Residuals & Information Flow in Transformers — Survey Completo

> **Insight centrale**: il residual fisso `x + f(x)` e' un bottleneck. Renderlo learned/dynamic/gated da guadagni enormi in compute effettiva per quasi zero overhead. Il campo sta convergendo su questo.

---

## TIER 1: Direttamente Applicabili a Speedruns

### 1. Attention Residuals (Kimi/Moonshot AI, Mar 2026)
- **Paper**: [arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031) | **Code**: [github.com/MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- **Idea**: Sostituisce l'accumulo residuale fisso con **softmax attention sugli output di tutti i layer precedenti** (attention depth-wise). Block AttnRes partiziona i layer in blocchi per ridurre overhead.
- **Risultati**: Testato su Kimi Linear (48B tot / 3B attivi) su 1.4T tokens. Mitiga la diluizione Pre-Norm, output magnitudes piu' uniformi. **1.25x compute advantage** alla scala maggiore.
- **Analisi**: [Ziming Liu: When does Attention Residuals work?](https://kindxiaoming.github.io/blog/2026/attention-residual/)
- **Insight**: la depth-wise residual attention e' lo stesso operatore locale della sliding-window attention causale, ma scritto sulla profondita' invece che sulla sequenza ([arxiv.org/abs/2603.16039](https://arxiv.org/html/2603.16039))

### 2. DeepCrossAttention (Google, Feb 2025)
- **Paper**: [arxiv.org/abs/2502.06785](https://arxiv.org/abs/2502.06785) | ICML 2025 | **Code**: [lucidrains/deep-cross-attention](https://github.com/lucidrains/deep-cross-attention)
- **Idea**: Ogni layer fa cross-attention depth-wise su tutti gli output dei layer precedenti per decidere cosa portare avanti. Pesi learnable e input-dependent.
- **Risultati**: Stessa qualita' **fino a 3x piu' veloce** in training. Solo +0.2% parametri.

### 3. MUDDFormer (Caiyun AI, Feb 2025)
- **Paper**: [arxiv.org/abs/2502.12170](https://arxiv.org/abs/2502.12170) | ICML 2025 | **Code**: [github.com/Caiyun-AI/MUDDFormer](https://github.com/Caiyun-AI/MUDDFormer)
- **Idea**: Dense connections dinamiche per-posizione e per-stream (Q, K, V, residual separatamente). Ogni token ottiene i propri pesi di mixing cross-layer.
- **Risultati**: **MUDDPythia-2.8B = Pythia-6.9B** in perplexity, rivale Pythia-12B in 5-shot. Solo +0.23% params, +0.4% FLOPs. Scala a 42 layer senza diminishing returns (vs Transformer++ che plateau a 24).

### 4. nGPT (NVIDIA, Oct 2024)
- **Paper**: [arxiv.org/abs/2410.01131](https://arxiv.org/abs/2410.01131) | **Code**: [lucidrains/nGPT-pytorch](https://github.com/lucidrains/nGPT-pytorch)
- **Idea**: TUTTI i vettori (embeddings, pesi, hidden states) normalizzati a vivere sull'ipersfera. Ogni layer contribuisce un displacement sulla superficie. Le moltiplicazioni matriciali diventano cosine similarity.
- **Risultati**: **4x-20x meno step di training** per stessa accuracy (dipende dalla seq length).
- **Note**: Claim molto aggressivo, da verificare su scala speedrun.

### 5. Gated Attention (Qwen, May 2025)
- **Paper**: [arxiv.org/abs/2505.06708](https://arxiv.org/abs/2505.06708) | NeurIPS 2025 **Oral** | **Code**: [github.com/qiuzh20/gated_attention](https://github.com/qiuzh20/gated_attention)
- **Idea**: Gate sigmoid per-head dopo SDPA: `output = sigma(X @ W) * Attn(X)`. Aggiunge non-linearita' e sparsity.
- **Risultati**: Attention sink da 46.7% a 4.8%. Validato a 1.7B dense e 15B MoE su 3.5T tokens. **Adottato in Qwen3-Next production.**
- **GIA' IN MODDED-NANOGPT** come sigmoid gate sulle attention heads.

### 6. Value Residual Learning (Oct 2024)
- **Paper**: [arxiv.org/abs/2410.17897](https://arxiv.org/abs/2410.17897) | ICLR 2025
- **Idea**: Residual connections sulle **value projections** nell'attention (non solo hidden state). ResFormer (residual cross-layer sui values) e SVFormer (tutti i layer condividono i value del primo layer → dimezza KV cache).
- **Risultati**: Matcha baseline con **16.1% meno parametri e 20.3% meno dati**.
- **GIA' IN MODDED-NANOGPT** come "value embeddings".

### 7. Hyper-Connections (ByteDance, Sep 2024) + mHC (DeepSeek, Dec 2025)
- **HC**: [arxiv.org/abs/2409.19606](https://arxiv.org/abs/2409.19606) | ICLR 2025
- **mHC**: [arxiv.org/abs/2512.24880](https://arxiv.org/abs/2512.24880)
- **Idea HC**: Espande il singolo residual stream in **n stream paralleli** con interazione learnable. Risolve il seesaw Pre-LN/Post-LN.
- **Idea mHC**: Fix stabilita' di HC vincolando le mixing matrices al **Birkhoff Polytope** (matrici doubly stochastic via Sinkhorn-Knopp). Ripristina la proprieta' identity mapping.
- **Risultati mHC**: Testato a 3B, 9B, 27B. Solo 6-7% overhead. Loss migliore, reasoning migliore.
- **IN MODDED-NANOGPT**: "Partitioned Hyperconnections" (Record #73, 1.49 min, Feb 2026).

### 8. HybridNorm (Mar 2025)
- **Paper**: [arxiv.org/abs/2503.04598](https://arxiv.org/abs/2503.04598) | NeurIPS 2025 | **Code**: [github.com/BryceZhuo/HybridNorm](https://github.com/BryceZhuo/HybridNorm)
- **Idea**: QKV normalization in attention + Post-Norm in FFN. Combina i punti di forza di entrambe le strategie di placement.
- **Risultati**: **1.4x speedup convergenza** rispetto a Pre-Norm standard.

---

## TIER 2: Fondamentali / Architetturali

### 9. DenseFormer (EPFL, Feb 2024)
- **Paper**: [arxiv.org/abs/2402.02622](https://arxiv.org/abs/2402.02622) | NeurIPS 2024
- **Idea**: Dopo ogni transformer block, media pesata learnable di TUTTI gli output precedenti (incluso input embedding). Solo poche migliaia di scalari extra anche a 100B.
- **Risultati**: Perplexity migliore a pari model size. Antenato di MUDDFormer/DCA.

### 10. Differential Transformer (Microsoft, Oct 2024)
- **Paper**: [arxiv.org/abs/2410.05258](https://arxiv.org/abs/2410.05258) | ICLR 2025
- **Idea**: Attention = differenza tra due softmax separate, come un amplificatore differenziale. Cancella il noise, promuove attention sparse e focalizzata.
- **Risultati**: Supera Transformer standard a tutte le scale. Forti guadagni su long-context, key retrieval, riduzione hallucination.

### 11. ResiDual (Microsoft, Apr 2023)
- **Paper**: [arxiv.org/abs/2304.14802](https://arxiv.org/abs/2304.14802)
- **Idea**: Due stream residuali paralleli: uno Post-LN (diversita' rappresentazione), uno Pre-LN (gradient flow). Fusi in output.
- **Risultati**: Supera sia Post-LN che Pre-LN a tutte le scale.

### 12. DeepNet (Microsoft, 2022)
- **Paper**: [arxiv.org/abs/2203.00555](https://arxiv.org/abs/2203.00555) | IEEE TPAMI 2024
- **Idea**: DeepNorm: scaling factor alpha sui residuali + inizializzazione speciale. Combina performance Post-LN con stabilita' Pre-LN.
- **Risultati**: Scala a 1000 layer (2500 sub-layers).

### 13. ReZero (2020/2021)
- **Paper**: [arxiv.org/abs/2003.04887](https://arxiv.org/abs/2003.04887) | UAI 2021 | **Code**: [github.com/majumderb/rezero](https://github.com/majumderb/rezero)
- **Idea**: `x + alpha * F(x)` con alpha scalare learnable inizializzato a 0.
- **Risultati**: 56% convergenza piu' veloce. Abilita 120 layer.
- **FONDAMENTALE**: i lambda scalars in modded-nanogpt sono essenzialmente questo.

### 14. Post-LN Is Back / Keel (ByteDance, Jan 2026)
- **Paper**: [arxiv.org/abs/2601.19895](https://arxiv.org/abs/2601.19895)
- **Idea**: Riesuma Post-LN sostituendo il residual path con **Highway-style connection** (gating learnable tra identity e transform).
- **Risultati**: Scala oltre 1000 layer, perplexity costantemente migliore di Pre-LN.

### 15. Peri-LN (Feb 2025)
- **Paper**: [arxiv.org/abs/2502.02732](https://arxiv.org/abs/2502.02732) | ICML 2025
- **Idea**: LayerNorm sia prima che dopo ogni sub-layer. Usato da Gemma e OLMo.
- **Risultati**: Varianza piu' bilanciata, gradienti piu' stabili fino a 3.2B.

### 16. sigmaReparam (Apple, 2023)
- **Paper**: [arxiv.org/abs/2303.06296](https://arxiv.org/abs/2303.06296) | ICML 2023 | **Code**: [github.com/apple/ml-sigma-reparam](https://github.com/apple/ml-sigma-reparam)
- **Idea**: Spectral normalization + scalare learnable su tutti i linear layers. Previene entropy collapse nell'attention.
- **Risultati**: Training ViT senza warmup, weight decay, LayerNorm, o adaptive optimizers.

### 17. ProRes (Mar 2026)
- **Paper**: [arxiv.org/abs/2603.05369](https://arxiv.org/abs/2603.05369)
- **Idea**: **Trick gratuito**: warm up scalari residuali da 0 a 1, layer piu' profondi piu' lenti. Zero costo a inference.
- **Risultati**: Migliora convergenza senza nessun overhead.

---

## TIER 3: Attention Sinks & Diagnostica

### 18. Unified View of Attention/Residual Sinks (Qwen, Jan 2026)
- **Paper**: [arxiv.org/abs/2601.22966](https://arxiv.org/abs/2601.22966)
- **Idea**: Attention sinks e residual sinks (outlier massicci) servono come fattori di rescaling. Possono essere assorbiti in parametri learnable o mitigati con gated rescaling.
- **Risultati**: +2 punti medi performance, +1.2 punti sotto W4A4 quantization.

### 19. Attention Sinks Induce Gradient Sinks (Mar 2026)
- **Paper**: [arxiv.org/abs/2603.17771](https://arxiv.org/abs/2603.17771)
- **Idea**: Attention sinks causano gradienti concentrati ("gradient sinks") che guidano attivazioni massive. Introduce **V-scale** per sopprimerle.

### 20. Attention Sinks Are Provably Necessary in Softmax (Mar 2026)
- **Paper**: [arxiv.org/abs/2603.11487](https://arxiv.org/abs/2603.11487)
- **Idea**: La normalizzazione softmax forza i sink. ReLU attention li elimina completamente.

---

## TIER 4: Routing & Adaptive Depth

### 21. Mixture-of-Depths (DeepMind, Apr 2024)
- **Paper**: [arxiv.org/abs/2404.02258](https://arxiv.org/abs/2404.02258)
- **Idea**: Router decide per-token se applicare il full transformer block o solo passare il residual. Budget statico di compute per layer.
- **Risultati**: Matcha baseline a pari FLOPS, **50%+ piu' veloce a inference**.

### 22. Inner Thinking Transformer (ACL 2025)
- **Paper**: [arxiv.org/abs/2502.13842](https://arxiv.org/abs/2502.13842)
- **Idea**: Layer come "thinking steps" con routing adattivo. Token difficili = piu' compute, facili = meno.
- **Risultati**: 162M raggiunge 96.5% delle performance di 466M. 43.2% meno dati di training.

### 23. Always Skip Attention (2025)
- **Paper**: [arxiv.org/abs/2505.01996](https://arxiv.org/abs/2505.01996)
- **Idea**: Rimuovere skip connections dall'attention e' catastrofico. Rimuoverle da FFN ha impatto modesto. **Le attention skip connections sono quelle critiche.**

### 24. Simplifying Transformer Blocks (ICLR 2024)
- **Paper**: [arxiv.org/abs/2311.01906](https://arxiv.org/abs/2311.01906)
- **Idea**: Molti componenti (skip, value/projection matrices, ordinamento sequenziale, normalizzazione) possono essere rimossi. Parallel attention+MLP funziona.
- **Risultati**: 16% throughput training piu' veloce.

---

## Cosa Usa Gia' Modded-NanoGPT (stato attuale)

Tutte queste tecniche residuali sono gia' nel codice speedrun simultaneamente:

| Tecnica | Origine Paper | Record # | Contributo |
|---|---|---|---|
| **X0 mixin** | (empirico) | Early | Embeddings originali mixati in ogni layer. Lambda ~45-50% nei layer finali |
| **UNet skip connections** | (empirico) | #11 (7.2min) | Layer 2->11, 4->10, 6->9. Lambda ~0.3-0.5 |
| **Value embeddings** | Value Residual Learning | #14 (4.41min) | Embeddings separati aggiunti ai values. ~45% contributo al layer 15 |
| **Sigmoid attention gates** | Gated Attention | - | Gate per-head sui primi 12 dim del residual stream |
| **Smear (token shift)** | (empirico) | #34 (2.55min) | `x[1:] += 0.07 * x[:-1]` |
| **Backout** | (empirico) | #40 (2.36min) | Sottrae frazione del residual mid-layer prima della head finale |
| **Partitioned HC** | Hyper-Connections | #73 (1.49min) | Multi-stream residuals partitioned |

**Insight chiave dall'analisi dei lambda** ([snimu's blog](https://snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html)): la profondita' effettiva e' molto minore di quella architetturale. I layer centrali vengono skippati 15-25%. Il modello calcola "la differenza tra input e target embeddings" piuttosto che trasformazioni profonde.

---

## Blog & Risorse Extra

- [LessWrong: How the NanoGPT Speedrun WR dropped 20% in 3 months](https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months)
- [snimu: Lambda analysis](https://snimu.github.io/2025/08/11/modded-nanogpt-lambdas.html)
- [snimu: Value embeddings record](https://snimu.github.io/2025/10/07/modded-nanogpt-value-embeddings.html)
- [snimu: Decoupling embedding size](https://snimu.github.io/2025/10/10/modded-nanogpt-x0.html)
- [Tyler Romero: NanoGPT Speedrun Worklog](https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/)
- [Speedrunning ideas discussion](https://github.com/KellerJordan/modded-nanogpt/discussions/23)
- [Ziming Liu: When does Attention Residuals work?](https://kindxiaoming.github.io/blog/2026/attention-residual/)

---

## Le Nostre Idee / Direzioni da Esplorare

1. **MUDDFormer in nanogpt**: il gain di 1.8-2.4x compute equivalente con +0.23% params sembra troppo buono. Vale la pena implementarlo nella speedrun. Il codice c'e' gia'.

2. **ProRes e' gratis**: warm up dei residual scalars da 0 a 1 durante training. Zero costo inference. Da provare subito.

3. **Combinare DCA + Gated Attention**: DeepCrossAttention per il mixing cross-layer + sigmoid gates per sparsificare l'attention. Nessuno l'ha ancora testato insieme.

4. **Attention Residuals blockwise nella speedrun**: la versione Block AttnRes di Kimi e' abbastanza leggera. Il fatto che depth-wise attention = sliding window attention sulla depth e' un'osservazione profonda — potrebbe esserci un kernel efficiente.

5. **Il pattern "il modello calcola la differenza"**: se il modello alla fine fa solo `embeddings + delta`, forse possiamo rendere esplicito questo inductive bias. Una architettura che computa esplicitamente il delta potrebbe convergere molto piu' veloce.
