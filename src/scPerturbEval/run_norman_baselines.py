from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess_norman_scdfm import load_preprocessed_norman


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_mean(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("Cannot compute mean of empty matrix.")
    return X.mean(axis=0)


def _safe_std(X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        raise ValueError("Cannot compute std of empty matrix.")
    s = X.std(axis=0)
    s[np.isnan(s)] = 0.0
    return s


def _tokenize_condition(cond: str) -> list[str]:
    return [t.strip() for t in cond.split("+") if t.strip() and t.strip().lower() != "control"]


def _build_condition_vocab(conditions: list[str]) -> list[str]:
    toks: set[str] = set()
    for c in conditions:
        toks.update(_tokenize_condition(c))
    return sorted(toks)


def _condition_to_multihot(cond: str, token_to_idx: dict[str, int]) -> np.ndarray:
    x = np.zeros(len(token_to_idx), dtype=np.float32)
    for t in _tokenize_condition(cond):
        if t in token_to_idx:
            x[token_to_idx[t]] = 1.0
    return x


def _sample_controls(
    control_pool: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = rng.choice(control_pool.shape[0], size=n, replace=True)
    return control_pool[idx]


def _paired_train_data(
    train_x: np.ndarray,
    train_obs: pd.DataFrame,
    rng: np.random.Generator,
    covariate_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctrl_mask = train_obs["is_control"].to_numpy().astype(bool)
    pert_mask = ~ctrl_mask
    ctrl_idx = np.where(ctrl_mask)[0]
    pert_idx = np.where(pert_mask)[0]
    if len(ctrl_idx) == 0 or len(pert_idx) == 0:
        raise ValueError("Need both control and perturbed train cells.")

    if covariate_col is None:
        matched_ctrl_idx = rng.choice(ctrl_idx, size=len(pert_idx), replace=True)
    else:
        cov = train_obs[covariate_col].astype(str).to_numpy()
        matched_ctrl_idx = np.zeros(len(pert_idx), dtype=int)
        for k, pi in enumerate(pert_idx):
            pool = ctrl_idx[cov[ctrl_idx] == cov[pi]]
            if len(pool) == 0:
                pool = ctrl_idx
            matched_ctrl_idx[k] = int(rng.choice(pool))

    x_ctrl = train_x[matched_ctrl_idx].astype(np.float32)
    y_pert = train_x[pert_idx].astype(np.float32)
    cond = train_obs.iloc[pert_idx]["condition"].astype(str).to_numpy()
    return x_ctrl, y_pert, cond


class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc2(self.fc1(x)))


class LatentAdditiveModel(nn.Module):
    # perturbench-style formula: y = softplus(Dec(Enc(c) + PertEnc(p)))
    def __init__(self, n_genes: int, n_perts: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.gene_encoder = nn.Sequential(nn.Linear(n_genes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.pert_encoder = nn.Sequential(nn.Linear(n_perts, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes))

    def forward(self, control_expr: torch.Tensor, pert: torch.Tensor) -> torch.Tensor:
        zc = self.gene_encoder(control_expr)
        zp = self.pert_encoder(pert)
        return F.softplus(self.decoder(zc + zp))


class DecoderOnlyModel(nn.Module):
    # perturbench-style formula: y = softplus(Dec(p))
    def __init__(self, n_genes: int, n_perts: int, hidden_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(n_perts, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_genes))

    def forward(self, pert: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.decoder(pert))


def _train_torch_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    steps: int,
    lr: float,
    seed: int,
    early_stopping: bool,
    val_fraction: float,
    patience: int,
    min_delta: float,
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_t_all = torch.tensor(x_train, dtype=torch.float32)
    y_t_all = torch.tensor(y_train, dtype=torch.float32)

    use_es = early_stopping and x_train.shape[0] >= 3 and val_fraction > 0.0
    if use_es:
        n = x_train.shape[0]
        n_val = max(1, int(round(n * val_fraction)))
        n_val = min(n_val, n - 1)
        perm = np.random.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        x_tr = x_t_all[tr_idx]
        y_tr = y_t_all[tr_idx]
        x_val = x_t_all[val_idx]
        y_val = y_t_all[val_idx]
        best_state = None
        best_val = float("inf")
        bad_steps = 0
    else:
        x_tr = x_t_all
        y_tr = y_t_all
        x_val = None
        y_val = None

    model.train()
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()
        if use_es:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)  # type: ignore[arg-type]
                val_loss = float(criterion(val_pred, y_val).item())  # type: ignore[arg-type]
            model.train()
            if val_loss < (best_val - min_delta):
                best_val = val_loss
                bad_steps = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad_steps += 1
                if bad_steps >= patience:
                    break

    if use_es and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_baseline(
    processed_dir: Path,
    baseline: str,
    out_dir: Path,
    *,
    steps: int = 5000,
    hidden_dim: int = 1024,
    lr: float = 1e-3,
    seed: int = 42,
    early_stopping: bool = True,
    val_fraction: float = 0.2,
    patience: int = 20,
    min_delta: float = 1e-4,
    covariate_col: str | None = None,
    latent_dim: int = 64,
) -> dict[str, Path]:
    _adata_all, adata_train, adata_test, _meta = load_preprocessed_norman(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shared_genes = adata_train.var_names.intersection(adata_test.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between preprocessed train and test data.")

    adata_train = adata_train[:, shared_genes].copy()
    adata_test = adata_test[:, shared_genes].copy()

    train_x = _to_dense(adata_train.X).astype(np.float32)
    test_x = _to_dense(adata_test.X).astype(np.float32)
    train_obs = adata_train.obs.copy()
    test_obs = adata_test.obs.copy()

    for col in ["condition", "mode", "is_control"]:
        if col not in train_obs.columns or col not in test_obs.columns:
            raise ValueError(f"Expected '{col}' in both train/test .obs")

    eval_mask = test_obs["mode"].astype(str).to_numpy() == "test"
    eval_obs = test_obs.loc[eval_mask].copy()
    eval_x = test_x[eval_mask]
    if eval_x.shape[0] == 0:
        raise ValueError("No test perturbation cells (mode=='test') found in test data.")

    test_ctrl = test_x[test_obs["is_control"].to_numpy().astype(bool)]
    train_ctrl = train_x[train_obs["is_control"].to_numpy().astype(bool)]
    if len(test_ctrl) == 0 or len(train_ctrl) == 0:
        raise ValueError("Need control cells in both train and test data.")

    rng = np.random.default_rng(seed)
    pred_x = np.zeros_like(eval_x, dtype=np.float32)

    # scPerturBench-style control baseline: use control-state statistics and sample synthetic cells.
    if baseline == "control":
        mu = _safe_mean(train_ctrl)
        sigma = _safe_std(test_ctrl)
        for cond in eval_obs["condition"].astype(str).unique():
            idx = np.where(eval_obs["condition"].astype(str).to_numpy() == cond)[0]
            noise = rng.normal(loc=0.0, scale=1.0, size=(len(idx), mu.shape[0])).astype(np.float32)
            pred_x[idx] = mu[None, :] + noise * sigma[None, :]

    # scPerturBench trainMean/controlMean + additive combo style.
    elif baseline == "global_delta_additive":
        non_control = train_obs[~train_obs["is_control"].to_numpy().astype(bool)]
        train_pert_x = train_x[~train_obs["is_control"].to_numpy().astype(bool)]
        if len(non_control) == 0:
            raise ValueError("No non-control train cells found.")

        cond_to_mean = {
            cond: _safe_mean(train_pert_x[non_control["condition"].astype(str).to_numpy() == cond])
            for cond in np.unique(non_control["condition"].astype(str).to_numpy())
        }
        single_means: dict[str, np.ndarray] = {}
        for cond, mu in cond_to_mean.items():
            toks = _tokenize_condition(cond)
            if len(toks) == 1:
                single_means[toks[0]] = mu

        mean_all = np.mean(np.stack(list(cond_to_mean.values()), axis=0), axis=0)
        ctrl_mu = _safe_mean(train_ctrl)
        sigma = _safe_std(test_ctrl)

        for cond in eval_obs["condition"].astype(str).unique():
            toks = _tokenize_condition(cond)
            if len(toks) <= 1:
                pred_mu = cond_to_mean.get(cond, mean_all)
            else:
                parts = [single_means.get(t, mean_all) for t in toks]
                pred_mu = np.sum(np.stack(parts, axis=0), axis=0) - (len(parts) - 1) * ctrl_mu
            idx = np.where(eval_obs["condition"].astype(str).to_numpy() == cond)[0]
            noise = rng.normal(loc=0.0, scale=1.0, size=(len(idx), pred_mu.shape[0])).astype(np.float32)
            pred_x[idx] = pred_mu[None, :] + noise * sigma[None, :]

    # scPerturBench baseReg style: paired control->pert regression with PCA + Ridge.
    elif baseline == "linear":
        x_ctrl_train, y_pert_train, cond_train = _paired_train_data(train_x, train_obs, rng, covariate_col)
        eval_cond = eval_obs["condition"].astype(str).to_numpy()
        eval_control = _sample_controls(test_ctrl, len(eval_obs), rng)

        for cond in np.unique(eval_cond):
            tr_mask = cond_train == cond
            te_mask = eval_cond == cond
            if tr_mask.sum() >= 3:
                n_comp = int(min(100, tr_mask.sum(), x_ctrl_train.shape[1]))
                pca = PCA(n_components=n_comp, random_state=seed)
                xtr_p = pca.fit_transform(x_ctrl_train[tr_mask])
                xte_p = pca.transform(eval_control[te_mask])
                reg = Ridge(random_state=seed)
                reg.fit(xtr_p, y_pert_train[tr_mask])
                pred_x[te_mask] = reg.predict(xte_p).astype(np.float32)
            else:
                pred_x[te_mask] = _safe_mean(train_ctrl)[None, :]

    # scPerturBench baseMLP style: one-hidden-layer mapping control->perturbed expression.
    elif baseline == "one_layer_mlp":
        x_ctrl_train, y_pert_train, _cond_train = _paired_train_data(train_x, train_obs, rng, covariate_col)
        model = OneHiddenLayerMLP(input_dim=x_ctrl_train.shape[1], hidden_dim=hidden_dim, output_dim=y_pert_train.shape[1])
        model = _train_torch_regressor(
            model,
            x_ctrl_train,
            y_pert_train,
            steps=steps,
            lr=lr,
            seed=seed,
            early_stopping=early_stopping,
            val_fraction=val_fraction,
            patience=patience,
            min_delta=min_delta,
        )
        eval_control = _sample_controls(test_ctrl, len(eval_obs), rng)
        with torch.no_grad():
            pred_x = model(torch.tensor(eval_control, dtype=torch.float32)).cpu().numpy().astype(np.float32)

    elif baseline in {"latent_additive", "decoder_only"}:
        x_ctrl_train, y_pert_train, cond_train = _paired_train_data(train_x, train_obs, rng, covariate_col)
        train_conditions = sorted(np.unique(cond_train).tolist())
        vocab = _build_condition_vocab(train_conditions)
        if len(vocab) == 0:
            raise ValueError(f"No perturbation tokens found for {baseline}.")
        token_to_idx = {t: i for i, t in enumerate(vocab)}

        p_train = np.vstack([_condition_to_multihot(c, token_to_idx) for c in cond_train]).astype(np.float32)
        p_eval = np.vstack([_condition_to_multihot(c, token_to_idx) for c in eval_obs["condition"].astype(str).to_numpy()]).astype(np.float32)

        if baseline == "latent_additive":
            model = LatentAdditiveModel(n_genes=train_x.shape[1], n_perts=len(token_to_idx), hidden_dim=hidden_dim, latent_dim=latent_dim)

            class _Wrap(nn.Module):
                def __init__(self, m: nn.Module, p: np.ndarray):
                    super().__init__()
                    self.m = m
                    self.p = torch.tensor(p, dtype=torch.float32)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self.m(x, self.p.to(x.device))

            wrapped = _Wrap(model, p_train)
            wrapped = _train_torch_regressor(
                wrapped,
                x_ctrl_train,
                y_pert_train,
                steps=steps,
                lr=lr,
                seed=seed,
                # Keep x/p row alignment intact for latent_additive.
                # Splitting inside the generic trainer breaks this alignment.
                early_stopping=False,
                val_fraction=0.0,
                patience=patience,
                min_delta=min_delta,
            )
            eval_control = _sample_controls(test_ctrl, len(eval_obs), rng)
            with torch.no_grad():
                pred_x = model(
                    torch.tensor(eval_control, dtype=torch.float32),
                    torch.tensor(p_eval, dtype=torch.float32),
                ).cpu().numpy().astype(np.float32)

        else:
            model = DecoderOnlyModel(n_genes=train_x.shape[1], n_perts=len(token_to_idx), hidden_dim=hidden_dim)
            model = _train_torch_regressor(
                model,
                p_train,
                y_pert_train,
                steps=steps,
                lr=lr,
                seed=seed,
                early_stopping=early_stopping,
                val_fraction=val_fraction,
                patience=patience,
                min_delta=min_delta,
            )
            with torch.no_grad():
                pred_x = model(torch.tensor(p_eval, dtype=torch.float32)).cpu().numpy().astype(np.float32)

    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    # Include real control rows in outputs (scDFM/scPerturBench-style export):
    # control predictions are copied from observed control expression.
    eval_idx = np.where(eval_mask)[0]
    ctrl_idx = np.where(test_obs["is_control"].to_numpy().astype(bool))[0]
    ctrl_only_idx = np.setdiff1d(ctrl_idx, eval_idx)
    if ctrl_only_idx.shape[0] > 0:
        ctrl_obs = test_obs.iloc[ctrl_only_idx].copy()
        ctrl_x = test_x[ctrl_only_idx].astype(np.float32)
        final_obs = pd.concat([ctrl_obs, eval_obs], axis=0)
        final_x = np.vstack([ctrl_x, eval_x]).astype(np.float32)
        final_pred = np.vstack([ctrl_x, pred_x]).astype(np.float32)
    else:
        final_obs = eval_obs.copy()
        final_x = eval_x.astype(np.float32)
        final_pred = pred_x.astype(np.float32)

    rows = []
    cond_arr = final_obs["condition"].astype(str).to_numpy()
    for cond in np.unique(cond_arr):
        idx = np.where(cond_arr == cond)[0]
        real_c = final_x[idx]
        pred_c = final_pred[idx]
        real_m = _safe_mean(real_c)
        pred_m = _safe_mean(pred_c)
        corr = pearsonr(real_m, pred_m)[0]
        rows.append(
            {
                "condition": cond,
                "n_cells": int(len(idx)),
                "rmse": float(np.sqrt(mean_squared_error(real_m, pred_m))),
                "mae": float(mean_absolute_error(real_m, pred_m)),
                "pearson": float(corr) if not np.isnan(corr) else np.nan,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)

    pred_adata = ad.AnnData(X=final_pred, obs=final_obs.copy(), var=adata_test.var.copy())
    real_adata = ad.AnnData(X=final_x, obs=final_obs.copy(), var=adata_test.var.copy())

    metrics_path = out_dir / f"{baseline}_metrics.csv"
    pred_path = out_dir / f"{baseline}_pred.h5ad"
    real_path = out_dir / f"{baseline}_real.h5ad"
    metrics_df.to_csv(metrics_path, index=False)
    pred_adata.write_h5ad(pred_path)
    real_adata.write_h5ad(real_path)

    return {"metrics": metrics_path, "pred": pred_path, "real": real_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load preprocessed Norman data and run baseline models.")
    parser.add_argument("--processed-dir", type=Path, required=True, help="Directory with preprocessed Norman files.")
    parser.add_argument(
        "--baseline",
        choices=["control", "global_delta_additive", "linear", "one_layer_mlp", "latent_additive", "decoder_only"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=5000, help="Training steps for neural baselines")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden size for neural baselines")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent size for latent_additive")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for neural baselines")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early-stopping", dest="early_stopping", action="store_true", help="Enable early stopping for neural baselines")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", help="Disable early stopping for neural baselines")
    parser.set_defaults(early_stopping=True)
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction for early stopping")
    parser.add_argument("--patience", type=int, default=20, help="Patience steps for early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation improvement for early stopping")
    parser.add_argument(
        "--covariate-col",
        default=None,
        help="Optional covariate obs column for matched-control sampling in paired baselines.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions and metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_baseline(
        processed_dir=args.processed_dir,
        baseline=args.baseline,
        out_dir=args.out_dir,
        steps=args.steps,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        seed=args.seed,
        early_stopping=args.early_stopping,
        val_fraction=args.val_fraction,
        patience=args.patience,
        min_delta=args.min_delta,
        covariate_col=args.covariate_col,
    )
    print("Wrote:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
