"""実行環境のセットアップに関するヘルパー関数。

このモジュールは「教材の本題ではないが、どのノートブックでも必要になる定型処理」を
まとめたものです。具体的には次の3つを担当します。

1. Google Colab / ローカルのどちらで動いているかの判定
2. プロジェクトのルートディレクトリと `output/` フォルダの解決
3. matplotlib の共通設定（図の見た目をシリーズ全体で揃える）

ノートブック側では ``from module.pgm_setup import setup_notebook`` のように
インポートして使います。ここに書かれた処理は機械学習の理解には不要なので、
ノートブック本文には展開していません。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GAS_NAMES = ["He", "H2", "O2", "N2", "CO2", "CH4"]
"""6種の対象ガス。データセットの列順・モデルの出力順はすべてこの順序に従う。"""

RANDOM_STATE = 42
"""シリーズ全体で共通の乱数シード。"""


def is_colab() -> bool:
    """Google Colab 上で実行されているかどうかを返す。"""
    return "google.colab" in sys.modules


def find_project_root(start: Path | None = None) -> Path:
    """`code/datasets/` を含むディレクトリを探索してプロジェクトルートを返す。

    ノートブックをリポジトリ直下で開いた場合も、Colab で `git clone` した直後に
    別階層から開いた場合も同じコードが動くようにするための処理。
    """
    candidates = []
    if start is not None:
        candidates.append(Path(start))
    candidates.append(Path.cwd())
    candidates.extend(Path.cwd().parents)
    candidates.append(Path.cwd() / "PolymerGasMembraneML")

    for base in candidates:
        if (base / "code" / "datasets" / "datasetA_imputed_all.csv").exists():
            return base.resolve()
    raise FileNotFoundError(
        "code/datasets/datasetA_imputed_all.csv が見つかりません。"
        "PolymerGasMembraneML リポジトリの直下でノートブックを開いてください。"
    )


def configure_matplotlib() -> None:
    """図の見た目をシリーズ全体で統一する。日本語フォントがあれば使用する。"""
    import matplotlib
    import matplotlib.pyplot as plt

    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.axisbelow"] = True

    try:  # 日本語フォントは任意。無ければ英語ラベルのみで問題なく動く。
        import matplotlib_fontja  # noqa: F401
    except ImportError:
        available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
        for name in ("IPAexGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo"):
            if name in available:
                plt.rcParams["font.family"] = name
                break


def setup_notebook(verbose: bool = True) -> dict:
    """ノートブック冒頭で1回だけ呼ぶ初期化関数。

    Returns
    -------
    dict
        ``ROOT``（プロジェクトルート）、``DATA_DIR``（`code/datasets/`）、
        ``MODEL_DIR``（`code/pretrained_models/`）、``OUTPUT_DIR``（生成物の保存先）
        を含む辞書。
    """
    root = find_project_root()
    paths = {
        "ROOT": root,
        "DATA_DIR": root / "code" / "datasets",
        "MODEL_DIR": root / "code" / "pretrained_models",
        "OUTPUT_DIR": root / "output",
    }
    paths["OUTPUT_DIR"].mkdir(exist_ok=True)

    # module/ を import 可能にする（Colab で cwd が異なる場合の保険）
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    configure_matplotlib()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if verbose:
        print(f"Environment      : {'Google Colab' if is_colab() else 'local'}")
        print(f"Project root     : {paths['ROOT']}")
        print(f"Dataset directory: {paths['DATA_DIR']}")
        print(f"Output directory : {paths['OUTPUT_DIR']}")
    return paths


def print_library_versions(names=("numpy", "pandas", "sklearn", "rdkit", "tensorflow", "shap")) -> None:
    """主要ライブラリのバージョンを表示する（再現性の記録用）。"""
    import importlib

    for name in names:
        try:
            module = importlib.import_module(name)
            print(f"{name:12s} {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            print(f"{name:12s} (not installed)")
