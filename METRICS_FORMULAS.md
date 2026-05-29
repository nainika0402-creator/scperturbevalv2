# scPerturbEval Metrics: Formulas and Implementation

This document describes how metrics are calculated in `scPerturbEval` (from `src/scPerturbEval/core_metrics.py` and `src/scPerturbEval/metrics.py`).

## Quick metric cheat sheet

| Metric | Better | What it captures (short) |
|---|---|---|
| `root_mean_squared_error` | lower | Mean-profile error magnitude per condition |
| `pearson_distance` | lower | 1 - Pearson correlation of real vs predicted mean profiles |
| `wasserstein` | lower | Distribution mismatch (gene-wise Wasserstein, averaged) |
| `mmd` | lower | Kernel two-sample distance between predicted and real cell distributions |
| `pcc_delta` | higher | Correlation of perturbation effects vs control (`Delta_real` vs `Delta_pred`) |
| `top_deg_recall` | higher | Fraction of top-N real DEGs recovered by top-N predicted DEGs |
| `deg_direction_agreement` | higher | Sign agreement of logFC on overlapping selected DEGs |
| `deg_spearman_lfc` | higher | Rank correlation of logFC on selected real DEGs |
| `pds_cosine` | higher | How highly each condition’s true effect is ranked by predicted effect similarity |
| `matrix_distance` | lower | Difference between condition-condition cosine structure (pred vs real) |
| `wmse` | lower | DEG-weighted mean-squared error on condition mean profile |
| `pearson_delta_pert` | higher | Correlation of condition deltas from global perturbed centroid |
| `weighted_r2_delta` | higher | Weighted explained variance of centroid-referenced perturbation deltas |
| `pathway_nes_spearman` | higher | Pathway-level NES rank consistency (GSEA-based) |
| `pathway_topk_jaccard` | higher | Overlap of top-K pathways by absolute NES |

## Notation

For a given perturbation condition `c`:

- `X_real^c in R^{n_r x G}`: real cells x genes
- `X_pred^c in R^{n_p x G}`: predicted cells x genes
- `mu_real^c = mean(X_real^c, axis=0) in R^G`
- `mu_pred^c = mean(X_pred^c, axis=0) in R^G`

For control condition `ctrl` (when required):

- `mu_real^ctrl`, `mu_pred^ctrl`
- `Delta_real^c = mu_real^c - mu_real^ctrl`
- `Delta_pred^c = mu_pred^c - mu_pred^ctrl`

Most metrics are reported per condition (row-wise), then aggregated (global mean row).

## Core vector/distribution metrics

### 1) `root_mean_squared_error`
Per condition:

`RMSE(c) = sqrt((1/G) * sum_g (mu_real^c[g] - mu_pred^c[g])^2)`

### 2) `pearson_distance`
Per condition:

`pearson_distance(c) = 1 - corr(mu_real^c, mu_pred^c)`

### 3) `wasserstein`
Per condition, per gene 1D Wasserstein, then averaged:

`W_g(c) = W1(X_real^c[:, g], X_pred^c[:, g])`

`wasserstein(c) = (1/G) * sum_g W_g(c)`

### 4) `mmd`
Gaussian-kernel MMD^2 on cell distributions:

`MMD^2(c) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]`

with `x, x' ~ X_pred^c`, `y, y' ~ X_real^c`,
`k(a,b) = exp(-||a-b||^2 / (2*sigma^2))`.

`sigma` is chosen via median-distance heuristic on pooled samples.

## Control-referenced metrics (require `--control-label`)

### 5) `pcc_delta`
Per condition:

`pcc_delta(c) = corr(Delta_real^c, Delta_pred^c)`

### 6) `top_deg_recall`
By default (`deg_selection="topn"`), DEGs are selected separately in real and predicted
as top-N genes ranked by absolute logFC (condition vs control):

- `logFC_real[g] = log2((mu_real^c[g] + eps)/(mu_real^ctrl[g] + eps))`
- `logFC_pred[g] = log2((mu_pred^c[g] + eps)/(mu_pred^ctrl[g] + eps))`
- `S_real^c = topN(|logFC_real|)`, `S_pred^c = topN(|logFC_pred|)`, with `N=top_n_degs` (default 100)
- if fewer than `N` genes exist, use all genes

Legacy mode (`deg_selection="fdr"`) recovers prior behavior using t-test + Benjamini-Hochberg,
with `S_real^c`, `S_pred^c` defined by `FDR <= deg_fdr_threshold`.

`top_deg_recall(c) = |S_real^c ∩ S_pred^c| / |S_real^c|`

(if denominator is 0, returns NaN)

### 7) `deg_direction_agreement` (aka direction_agreement)
On overlap `S_real^c ∩ S_pred^c`:

`deg_direction_agreement(c) = mean( sign(logFC_real[g]) == sign(logFC_pred[g]) )`

over overlap genes; NaN if overlap is empty.

### 8) `deg_spearman_lfc`
On genes selected in real (`S_real^c`):

`deg_spearman_lfc(c) = SpearmanCorr(logFC_real[S_real^c], logFC_pred[S_real^c])`

NaN if `S_real^c` empty.

### 9) `pds_cosine`
For each condition `c`, compare predicted effect vector `Delta_pred^c` to all real effect vectors `{Delta_real^j}` using cosine distance.

Score is normalized rank of correct match `j=c`:

`pds_cosine(c) = 1 - rank_c / (N-1)`

where rank is 0 for best (closest) match; higher is better.

### 10) `matrix_distance`
Build cosine-similarity matrices across conditions:

`S_pred[i,j] = cos(Delta_pred^i, Delta_pred^j)`

`S_real[i,j] = cos(Delta_real^i, Delta_real^j)`

`D = S_pred - S_real`

Global score:

`matrix_distance = ||D||_F`

(Unnormalized Frobenius norm; lower is better. This matches PerturBench scale and is not comparable to older normalized `/N` runs.)

## Perturbation-centroid referenced metrics

For non-control conditions, define global real centroid:

`mu_all = (1/K) * sum_c mu_real^c`

and DEG-derived condition weights `w^c in R^G` (nonnegative, sum to 1), from condition-vs-rest absolute t-statistics.

### 11) `wmse`
Per condition:

`wmse(c) = sum_g w^c[g] * (mu_real^c[g] - mu_pred^c[g])^2`

### 12) `pearson_delta_pert`
Per condition:

`d_real^c = mu_real^c - mu_all`

`d_pred^c = mu_pred^c - mu_all`

`pearson_delta_pert(c) = corr(d_real^c, d_pred^c)`

### 13) `weighted_r2_delta` (weighted R²(Δ))
Per condition with `d_real^c, d_pred^c` as above and normalized weights `w^c`:

`dbar = sum_g w^c[g] * d_real^c[g]`

`res = sum_g w^c[g] * (d_real^c[g] - d_pred^c[g])^2`

`tot = sum_g w^c[g] * (d_real^c[g] - dbar)^2`

`weighted_r2_delta(c) = 1 - res/tot`

NaN if `tot = 0`.

## Pathway metrics (raw space only)

Let `delta_real = mu_real^c - ref`, `delta_pred = mu_pred^c - ref`, where `ref` is configured by `--pathway-reference`.

Run prerank GSEA on both ranked gene lists to get NES vectors.

### 14) `pathway_nes_spearman`

`pathway_nes_spearman(c) = SpearmanCorr(NES_real, NES_pred)`

with missing terms filled by 0 after union of pathway sets.

### 15) `pathway_topk_jaccard`
Top-K by absolute NES in each:

`pathway_topk_jaccard(c) = |TopK_real ∩ TopK_pred| / |TopK_real ∪ TopK_pred|`

## Spaces (`--space`)

Metrics are computed after optional transform:

- `raw`: use original aligned genes
- `pca`: fit PCA on pooled real+pred for each condition, evaluate in PCA space
- `deg`: evaluate only DEG-selected genes (by logFC threshold/top-N); fallback to full set if no genes pass

## Source files

- `src/scPerturbEval/core_metrics.py`
- `src/scPerturbEval/metrics.py`
- `src/scPerturbEval/evaluations.py`
