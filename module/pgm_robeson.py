"""Robeson プロット（透過係数 vs 選択性）の描画と、上限線からの距離の計算。

原著リポジトリ `code/visualizations.ipynb` の ``plotRobeson()`` を関数として
整理し直したものです。上限線の係数は原著の実装からそのまま引き写しています。

Robeson 上限とは
----------------
高分子膜には「透過性を上げると選択性が下がる」というトレードオフがあり、
両対数プロット上で右下がりの直線が経験的な性能限界を与える（Robeson 上限）。
直線は log10(alpha) = log10(beta) - n * log10(P_fast) の形で表され、
``(beta, n)`` は分離ペアと年代ごとに決まる。

膜の設計目標は、この直線より上（＝右上）に位置する材料を見つけること。
"""

from __future__ import annotations

import numpy as np

from .pgm_setup import GAS_NAMES

GAS_INDEX = {gas: i for i, gas in enumerate(GAS_NAMES)}

SEPARATIONS = {
    # 名前: (速いガス, 遅いガス, {年: (log10(beta), n)}, x軸範囲, y軸範囲)
    "O2/N2": {
        "fast": "O2",
        "slow": "N2",
        "bounds": {
            1991: (np.log10(9.2008), 0.1724),
            2008: (np.log10(12.148), 0.1765),
            2015: (np.log10(18.50), 0.1754),
        },
        "xlim": (-4, 7),
        "ylim": (-1, 2),
    },
    "CO2/CH4": {
        "fast": "CO2",
        "slow": "CH4",
        "bounds": {
            1991: (np.log10(197.81), 0.3807),
            2008: (np.log10(357.33), 0.3794),
            2019: (np.log10(1155.60), 0.4165),
        },
        "xlim": (-2, 7),
        "ylim": (-2, 4),
    },
    "CO2/N2": {
        "fast": "CO2",
        "slow": "N2",
        "bounds": {
            2008: (np.log10(30_967_000) / 2.888, 1 / 2.888),
            2019: (np.log10(755.58e6) / 3.409, 1 / 3.409),
        },
        "xlim": (-2, 7),
        "ylim": (-1, 3),
    },
    "H2/CO2": {
        "fast": "H2",
        "slow": "CO2",
        "bounds": {
            1991: (np.log10(1200) / 1.9363, 1 / 1.9363),
            2008: (np.log10(4515) / 2.302, 1 / 2.302),
        },
        "xlim": (-2, 7),
        "ylim": (-1.5, 2),
    },
}

LINE_STYLES = {1991: "-", 2008: "--", 2015: ":", 2019: ":"}


def upper_bound(separation: str, year: int, log_permeability):
    """指定した分離ペア・年代の Robeson 上限線の log10(選択性) を返す。"""
    log_beta, slope = SEPARATIONS[separation]["bounds"][year]
    return log_beta - slope * np.asarray(log_permeability, dtype=float)


def distance_above_bound(log_permeabilities, separation: str, year: int = 2008):
    """各サンプルが Robeson 上限線から縦方向にどれだけ上にあるかを返す。

    正なら上限を超えている。単位は log10(選択性)。
    `pretrained_models/DNN_BLR_fing/top_polymers.csv` の
    ``ONdist`` / ``CCdist`` / ``CNdist`` / ``HCdist`` 列と同じ定義。
    """
    log_permeabilities = np.asarray(log_permeabilities, dtype=float)
    config = SEPARATIONS[separation]
    fast = log_permeabilities[:, GAS_INDEX[config["fast"]]]
    slow = log_permeabilities[:, GAS_INDEX[config["slow"]]]
    return (fast - slow) - upper_bound(separation, year, fast)


def robeson_coordinates(log_permeabilities, separation: str):
    """Robeson プロットの (x, y) = (log10 透過係数, log10 選択性) を返す。"""
    log_permeabilities = np.asarray(log_permeabilities, dtype=float)
    config = SEPARATIONS[separation]
    fast = log_permeabilities[:, GAS_INDEX[config["fast"]]]
    slow = log_permeabilities[:, GAS_INDEX[config["slow"]]]
    return fast, fast - slow


def draw_upper_bounds(ax, separation: str) -> None:
    """指定した軸に Robeson 上限線を描き入れる。"""
    config = SEPARATIONS[separation]
    x_min, x_max = config["xlim"]
    grid = np.array([x_min, x_max])
    for year, _ in sorted(config["bounds"].items()):
        ax.plot(grid, upper_bound(separation, year, grid), LINE_STYLES.get(year, "-."),
                color="black", linewidth=1.2, label=f"{year} upper bound")


def plot_robeson_panel(ax, log_permeabilities, separation: str, label: str = None,
                       show_bounds: bool = True, **scatter_kwargs) -> None:
    """1枚の軸に Robeson プロットを描く。"""
    config = SEPARATIONS[separation]
    x, y = robeson_coordinates(log_permeabilities, separation)
    options = {"s": 8, "alpha": 0.4, "edgecolors": "none"}
    options.update(scatter_kwargs)
    ax.scatter(x, y, label=label, **options)
    if show_bounds:
        draw_upper_bounds(ax, separation)
    ax.set_xlim(config["xlim"])
    ax.set_ylim(config["ylim"])
    ax.set_xlabel(f"log10 P({config['fast']}) [Barrer]")
    ax.set_ylabel(f"log10 selectivity {separation}")
    ax.set_title(f"{separation} separation")


def plot_robeson_grid(datasets: dict, separations=None, figsize=(13, 10),
                      show_bounds: bool = True, colors=None):
    """4つの分離ペアについて Robeson プロットを 2x2 で描く。

    Parameters
    ----------
    datasets : dict
        ``{ラベル: 6ガスのlog10透過係数を並べた配列}``。描画順は辞書の順序に従う。
    """
    import matplotlib.pyplot as plt

    separations = separations or list(SEPARATIONS)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, separation in zip(axes.ravel(), separations):
        for i, (label, values) in enumerate(datasets.items()):
            kwargs = {}
            if colors is not None:
                kwargs["color"] = colors[i % len(colors)]
            plot_robeson_panel(ax, values, separation, label=label, show_bounds=False, **kwargs)
        if show_bounds:
            draw_upper_bounds(ax, separation)
        ax.legend(fontsize=8, loc="upper right", markerscale=2)
    fig.tight_layout()
    return fig, axes


def count_above_bound(log_permeabilities, year: int = 2008) -> "pd.DataFrame":
    """4つの分離ペアそれぞれについて、上限線を超えたサンプル数を数える。

    論文 Table 3 に対応する集計。
    """
    import pandas as pd

    rows = []
    for separation, config in SEPARATIONS.items():
        if year not in config["bounds"]:
            continue
        distance = distance_above_bound(log_permeabilities, separation, year)
        rows.append(
            {
                "separation": separation,
                f"n_above_{year}": int((distance > 0).sum()),
                "fraction": float((distance > 0).mean()),
                "max_distance": float(distance.max()),
            }
        )
    return pd.DataFrame(rows)
