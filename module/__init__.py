"""高分子ガス分離膜の機械学習セミナー教材のヘルパーモジュール群。

各ノートブックからは、たとえば次のようにインポートして使います。

>>> from module.pgm_setup import setup_notebook
>>> from module.pgm_data import load_dataset_a, get_targets

収録内容
--------
* :mod:`module.pgm_setup`    実行環境の判定・パス解決・図の共通設定
* :mod:`module.pgm_data`     データセットの読み込みと整形
* :mod:`module.pgm_features` RDKit 記述子・Morgan fingerprint の計算
* :mod:`module.pgm_models`   DNN アンサンブル / RF の構築・学習・読み込み
* :mod:`module.pgm_robeson`  Robeson プロットと上限線からの距離
* :mod:`module.pgm_shap`     SHAP 値の計算・集計・可視化

いずれも「教材の本題ではないが必要な定型処理」を切り出したものです。
機械学習の要点そのものはノートブック本文のコードセルに残してあります。
"""

__all__ = ["pgm_setup", "pgm_data", "pgm_features", "pgm_models", "pgm_robeson", "pgm_shap"]
