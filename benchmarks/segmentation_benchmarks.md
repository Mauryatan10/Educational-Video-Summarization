# Discourse Topic Segmentation Benchmarks



---

### Metric Definitions
* **Strict Match ($w=0$) $F_1$**: Exact sentence-index boundary match.
* **Standard Relaxed ($w=1$) $F_1$**: Greedy 1-to-1 match within $\pm 1$ sentence tolerance (primary selection metric).
* **Discourse Window ($w=2$) $F_1$**: Greedy 1-to-1 match within $\pm 2$ sentences tolerance.
* **Beeferman's $P_k$ ($\downarrow$) & Pevzner–Hearst WindowDiff $WD$ ($\downarrow$)**: Lower is better; measures probabilistic boundary error across sliding probe windows.


---

## 2. Master Comparative Table Across All Architectures (YT-Seg Multimodal Tier-1 Benchmark, $N = 94$)

Evaluated consistently across the **94 validated Tier-1 YT-Seg lectures** (`data/ytseg_val_100/manifest_100.json`, 1D-NMS radius $r = 6$):

| # | Architecture / Research Stage | Backbone | Modality & Fusion | Training Scale | Threshold ($\tau$) | Strict $F_1$ ($w=0$) | Relaxed $F_1$ ($w=1$) | Window 2 $F_1$ ($w=2$) | Precision ($w=1$) | Recall ($w=1$) |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **MiniSeg Baseline (Legacy)** | `all-MiniLM-L6-v2` | Text-only + BiLSTM/TransEnc | ytseg train | 0.80 | 26.40% | 39.50% | 46.13% | 34.80% | 52.10% |
| **2** | **ModernBERT-base + DBCA** | `ModernBERT-base` (148M) | Text-only + LoRA + DBCA | 600 docs | — | 28.12% | 42.10% | 50.91% | 37.70% | **60.50%** |
| **3** | **Unimodal Qwen 3.5 + DBCA** | `Qwen/Qwen3.5-0.8B` | Text-only + DBCA | 600 docs | 0.22 | 29.65% | 44.35% | 55.05% | 39.26% | 54.33% |
| **4** | **3-Stage Multimodal Qwen3-Embedding (BEST)** | **`Qwen/Qwen3-Embedding-0.6B`** | **Text + Qwen ViT + Stage-3 LoRA + Trimodal DBCA** | **592 docs** | **0.24** | **`39.16%`** | **`51.37%`** | 57.99% | **`55.05%`** | 50.94% |



---

