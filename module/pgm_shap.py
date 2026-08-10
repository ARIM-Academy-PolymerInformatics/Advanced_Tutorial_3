"""SHAP 値の計算・集計・可視化のヘルパー。

原著リポジトリの `code/step3.5_SHAP.py` と `code/visualizations.ipynb` の
``plotSHAP()`` に相当します。SHAP そのものの考え方は 04冊目の本文で説明し、
ここには「16 モデル分の SHAP 値を平均する」「上位特徴量を並べ替える」といった
定型処理だけを置いています。

原著からの変更点
----------------
* 原著は `shap.explainers._deep.Deep`（非公開パス）を直接呼んでいたが、
  公開 API の `shap.DeepExplainer` を使う。
* 背景データ（background）に学習データ全件を使うと計算量が大きいため、
  サンプル数を引数で制御できるようにした。既定は原著同様の全件。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pgm_setup import GAS_NAMES


def deep_shap_values(models, X, background=None, n_background: int | None = None,
                     random_state: int = 42, verbose: bool = True) -> np.ndarray:
    """DNN アンサンブルの SHAP 値を計算し、モデル間で平均する。

    Returns
    -------
    ndarray, shape (n_samples, n_features, n_gases)
        各サンプル・各特徴量が、各ガスの予測にどれだけ寄与したか。
    """
    import shap

    X = np.asarray(X, dtype="float64")
    if background is None:
        if n_background is not None and n_background < len(X):
            rng = np.random.default_rng(random_state)
            background = X[rng.choice(len(X), n_background, replace=False)]
        else:
            background = X

    total = None
    for i, model in enumerate(models):
        if verbose:
            print(f"computing SHAP values for model {i + 1}/{len(models)}")
        explainer = shap.DeepExplainer(model, background)
        values = np.asarray(explainer.shap_values(X, check_additivity=False))
        total = values if total is None else total + values
    return total / len(models)


def cached_deep_shap_values(models, X, cache_path, recompute: bool = False, **kwargs) -> np.ndarray:
    """:func:`deep_shap_values` の結果を ``.npy`` にキャッシュして再利用する。

    SHAP 値の計算は16モデル分で1分前後かかる。ノートブックを再実行するたびに
    待たされるのを避けるため、結果をファイルに保存しておく。
    キャッシュの形状が入力と合わない場合は自動的に再計算する。

    Parameters
    ----------
    cache_path : str or Path
        保存先（例: ``OUTPUT_DIR / 'shap_fing.npy'``）
    recompute : bool
        True ならキャッシュを無視して計算し直す。
    """
    from pathlib import Path

    cache_path = Path(cache_path)
    expected = (np.asarray(X).shape[0], np.asarray(X).shape[1], len(GAS_NAMES))
    if not recompute and cache_path.exists():
        values = np.load(cache_path)
        if values.shape == expected:
            print(f"cached SHAP values loaded from {cache_path.name}")
            return values
        print(f"cache shape mismatch ({values.shape} != {expected}); recomputing")

    values = deep_shap_values(models, X, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, values)
    print(f"SHAP values computed and cached to {cache_path.name}")
    return values


def tree_shap_values(model, X) -> np.ndarray:
    """ランダムフォレスト（マルチ出力）の SHAP 値を計算する。"""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(np.asarray(X, dtype="float64"))
    return np.asarray(values)


def mean_absolute_importance(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    """|SHAP| の平均を特徴量 x ガスの表にまとめる（論文 Fig.3A / Fig.4A に対応）。"""
    shap_values = np.asarray(shap_values)
    if shap_values.ndim != 3:
        raise ValueError("shap_values must have shape (n_samples, n_features, n_gases)")
    importance = np.abs(shap_values).mean(axis=0)  # (n_features, n_gases)
    frame = pd.DataFrame(importance, index=list(feature_names), columns=GAS_NAMES)
    frame["total"] = frame.sum(axis=1)
    return frame.sort_values("total", ascending=False)


def plot_importance_bars(importance: pd.DataFrame, top_n: int = 12, ax=None,
                         title: str = "Average SHAP importance"):
    """上位 top_n 特徴量の重要度をガス別の積み上げ棒グラフで描く。"""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    top = importance.head(top_n)[GAS_NAMES]
    top.plot(kind="bar", ax=ax, colormap="Set2", width=0.8)
    ax.set_ylabel("mean |SHAP value|")
    ax.set_xlabel("feature")
    ax.set_title(title)
    ax.legend(title="gas", fontsize=8, ncol=2)
    return ax


def signed_effect(shap_values: np.ndarray, X, feature_names, gas: str = "CH4") -> pd.DataFrame:
    """特徴量の値と SHAP 値の相関から、透過係数への寄与の向き（正/負）を判定する。

    論文 Fig.3C / Fig.4C の「赤＝透過性を上げる、青＝下げる」に対応する。
    相関が正なら「その特徴量が大きいほど透過係数が上がる」ことを意味する。
    """
    shap_values = np.asarray(shap_values)
    X = np.asarray(X, dtype="float64")
    gas_index = GAS_NAMES.index(gas)
    rows = []
    for j, name in enumerate(feature_names):
        feature = X[:, j]
        contribution = shap_values[:, j, gas_index]
        if np.std(feature) == 0 or np.std(contribution) == 0:
            correlation = np.nan
        else:
            correlation = float(np.corrcoef(feature, contribution)[0, 1])
        rows.append(
            {
                "feature": name,
                "correlation": correlation,
                "effect": "positive" if correlation > 0 else ("negative" if correlation < 0 else "n/a"),
                "mean_abs_shap": float(np.abs(contribution).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def plot_beeswarm(shap_values: np.ndarray, X, feature_names, gas: str = "CH4",
                  top_n: int = 12, show: bool = True):
    """指定ガスについて SHAP の beeswarm プロットを描く（論文 Fig.3B / Fig.4B）。"""
    import shap

    gas_index = GAS_NAMES.index(gas)
    return shap.summary_plot(
        np.asarray(shap_values)[:, :, gas_index],
        np.asarray(X, dtype="float64"),
        feature_names=[str(n) for n in feature_names],
        max_display=top_n,
        show=show,
    )
