"""データセットの読み込みに関するヘルパー関数。

原著リポジトリの `code/step3_train.py` などが各スクリプトで重複して書いていた
「CSVを読み、SMILESごとに平均を取り、補完済み透過係数の列を切り出す」処理を
1か所にまとめたものです。ノートブックでは処理の意味だけ説明し、実装は
このモジュールに置いています。

原著コードとの対応
------------------
* :func:`load_dataset_a` … `step3_train.py` 冒頭の `pd.read_csv` + `groupby('Smiles').mean()`
* :func:`get_targets`    … `step3_train.py` の `Y = DatasetA_grouped.iloc[:,-12:-6]`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .pgm_setup import GAS_NAMES

IMPUTATION_SUFFIX = {"BLR": "Bayesian", "ERT": "Etree"}
"""補完手法の識別子と、`datasetA_imputed_all.csv` の列名サフィックスの対応。"""


def load_dataset_a_raw(data_dir: Path) -> pd.DataFrame:
    """Dataset A（学習用）を生のまま読み込む。778 行（文献エントリ単位）。"""
    return pd.read_csv(Path(data_dir) / "datasetA_imputed_all.csv", index_col=0)


def load_dataset_a(data_dir: Path) -> pd.DataFrame:
    """Dataset A を SMILES 単位に集約した表を返す。353 行（ユニークな高分子）。

    同じ高分子について複数の文献値がある場合、原著は数値列の単純平均を取る。
    pandas の新しいバージョンでは非数値列があると `mean()` が例外を出すため、
    `numeric_only=True` を明示している（原著の pandas 0.24 では既定の挙動）。
    """
    raw = pd.read_csv(Path(data_dir) / "datasetA_imputed_all.csv")
    return raw.groupby("Smiles").mean(numeric_only=True).reset_index()


def get_targets(grouped: pd.DataFrame, imputation: str = "BLR") -> pd.DataFrame:
    """補完済みの log10 透過係数（6ガス分）を列名で取り出す。

    原著は `iloc[:, -12:-6]` という位置指定だったが、列の並び順の変更に弱いため
    列名指定に置き換えている（本プロジェクトのコーディング規約）。
    """
    if imputation not in IMPUTATION_SUFFIX:
        raise ValueError(f"imputation must be one of {list(IMPUTATION_SUFFIX)}")
    suffix = IMPUTATION_SUFFIX[imputation]
    columns = [f"log10_{gas}_{suffix}" for gas in GAS_NAMES]
    return grouped[columns]


def get_measured_targets(grouped: pd.DataFrame) -> pd.DataFrame:
    """補完前の（欠損を含む）log10 透過係数を返す。補完の効果を見るために使う。"""
    return grouped[[f"log10_{gas}" for gas in GAS_NAMES]]


def load_features(data_dir: Path, features: str = "fing") -> pd.DataFrame:
    """Dataset A について事前計算済みの化学特徴量を読み込む。

    Parameters
    ----------
    features : {'fing', 'desc'}
        'fing' は Morgan fingerprint with frequency（114 部分構造）、
        'desc' は RDKit 分子記述子（146 個）。
    """
    if features not in ("fing", "desc"):
        raise ValueError("features must be 'fing' or 'desc'")
    return pd.read_csv(Path(data_dir) / f"datasetAX_{features}.csv")


def load_screening_smiles(data_dir: Path, name: str, n_rows: int | None = None,
                          sample_n: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """スクリーニング用データセット（B / C / D）の SMILES を読み込む。

    Parameters
    ----------
    name : {'datasetB', 'datasetC_0', 'datasetD'}
    n_rows : int, optional
        先頭 n_rows 行だけを読む（動作確認用）。
    sample_n : int, optional
        全件を読んでから n 件を**無作為抽出**する。教材ではこちらを推奨する。
        Dataset C はジアミン成分ごとに並んでいるため、先頭から順に取ると
        化学的に偏った部分集合になってしまう。
    """
    frame = pd.read_csv(Path(data_dir) / f"{name}.csv", nrows=n_rows)
    if sample_n is not None and sample_n < len(frame):
        frame = frame.sample(n=sample_n, random_state=random_state).reset_index(drop=True)
    return frame


def summarize_missingness(grouped: pd.DataFrame) -> pd.DataFrame:
    """ガスごとの実測値の件数と欠損率をまとめた表を返す。"""
    measured = get_measured_targets(grouped)
    rows = []
    for gas in GAS_NAMES:
        column = measured[f"log10_{gas}"]
        rows.append(
            {
                "gas": gas,
                "n_measured": int(column.notna().sum()),
                "n_missing": int(column.isna().sum()),
                "missing_rate": float(column.isna().mean()),
                "min_log10P": float(np.nanmin(column)),
                "max_log10P": float(np.nanmax(column)),
            }
        )
    return pd.DataFrame(rows)
