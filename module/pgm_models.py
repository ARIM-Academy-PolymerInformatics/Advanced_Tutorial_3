"""DNN アンサンブル / ランダムフォレストの構築・学習・読み込み。

原著リポジトリの `code/DNN_functions.py` と `code/step3_train.py` を、
現行の TensorFlow / Keras 3 で動くように書き直したものです。
モデルの構造・ハイパーパラメータ・ブートストラップの手続きは原著と同一です。

原著からの変更点
----------------
1. **非公開 API の排除**
   原著の損失関数は `tensorflow.python.ops.math_ops` や
   `tensorflow.python.framework.ops.convert_to_tensor_v2` という
   TensorFlow の内部 API を使っており、TF 2.6 以降では動かない。
   公開 API のみを使う形に書き直した（数式は同一）。
2. **学習済みモデルの読み込み方法**
   同梱の `pretrained_models/` は TensorFlow 2.3 時代の SavedModel 形式で、
   Keras 3 では `load_model` で読めない（optimizer slot 変数の復元で失敗する）。
   本モジュールはチェックポイントから重み行列だけを取り出し、同じ構造の
   Keras モデルに流し込む方式を採る。この方式で原著の予測値を
   誤差 1e-15 以内で再現できることを 03冊目で検証する。
3. `os.chdir` による作業ディレクトリ書き換えの排除。

これらはいずれも「機械学習の理解」ではなく環境互換性の問題なので、
ノートブック本文には展開せずこのモジュールに置いています。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pgm_setup import GAS_NAMES, RANDOM_STATE

HIDDEN_UNITS = (64, 64, 32, 16, 8)
"""DNN の隠れ層のユニット数。論文 Methods 節「five hidden layers with 64, 64, 32, 16, and 8 nodes」。"""

DROPOUT_RATE = 0.1
EPOCHS = 50
BATCH_SIZE = 64
N_ENSEMBLE = 16
"""アンサンブルを構成する DNN の個数。論文「Sixteen independent models are trained」。"""

BOOTSTRAP_FRACTION = 0.8
"""各 DNN が使う訓練データの割合。論文「using 80% of the entries in the non-holdout set each time」。"""


# ------------------------------------------------------------------ 損失関数
def nan_mean_squared_error(y_true, y_pred):
    """欠損値（NaN）を無視する平均二乗誤差。

    補完していない生データで学習する場合に使う。NaN の位置の残差をゼロで置き換え、
    分母を「その行で観測されているガスの数」にすることで、欠損の多い行が
    不当に小さい損失にならないようにしている。

    補完済みデータ（BLR / ERT）で学習する場合は NaN が無いため、通常の MSE と一致する。
    """
    import tensorflow as tf

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    residual = y_true - y_pred
    residual = tf.where(tf.math.is_nan(residual), tf.zeros_like(residual), residual)
    n_observed = tf.reduce_sum(tf.cast(~tf.math.is_nan(y_true), y_pred.dtype), axis=-1)
    return tf.reduce_sum(tf.square(residual), axis=-1) / n_observed


# -------------------------------------------------------------- モデルの構築
def build_dnn(n_features: int, n_outputs: int = len(GAS_NAMES), seed: int | None = None):
    """論文と同じ構造のマルチタスク DNN を1つ作る（未学習）。"""
    import keras

    initializer_seed = {} if seed is None else {"kernel_initializer": keras.initializers.GlorotUniform(seed=seed)}
    layers = [keras.layers.Input(shape=(n_features,))]
    for units in HIDDEN_UNITS:
        layers.append(keras.layers.Dense(units, activation="relu", **initializer_seed))
    layers.append(keras.layers.Dropout(DROPOUT_RATE, seed=seed))
    layers.append(keras.layers.Dense(n_outputs, **initializer_seed))
    return keras.Sequential(layers)


def _dense_layers(model):
    import keras

    return [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]


# -------------------------------------------------- 学習済みモデルの読み込み
def load_pretrained_dnn(model_dir: Path, n_features: int):
    """TF 2.3 時代の SavedModel から重みだけを取り出し、Keras 3 のモデルに復元する。"""
    import tensorflow as tf

    reader = tf.train.load_checkpoint(str(Path(model_dir) / "variables" / "variables"))
    model = build_dnn(n_features)
    for i, layer in enumerate(_dense_layers(model)):
        kernel = reader.get_tensor(f"layer_with_weights-{i}/kernel/.ATTRIBUTES/VARIABLE_VALUE")
        bias = reader.get_tensor(f"layer_with_weights-{i}/bias/.ATTRIBUTES/VARIABLE_VALUE")
        layer.set_weights([kernel, bias])
    return model


def load_pretrained_ensemble(ensemble_dir: Path, n_features: int, n_models: int = N_ENSEMBLE):
    """`pretrained_models/DNN_BLR_fing/DNN_0` … `DNN_15` をまとめて読み込む。"""
    ensemble_dir = Path(ensemble_dir)
    return [load_pretrained_dnn(ensemble_dir / f"DNN_{i}", n_features) for i in range(n_models)]


# -------------------------------------------------------------------- 学習
def train_dnn_ensemble(X, y, n_models: int = N_ENSEMBLE, epochs: int = EPOCHS,
                       random_state: int = RANDOM_STATE, verbose: bool = True):
    """ブートストラップで DNN アンサンブルを学習する（論文 Methods「Ensembling」）。

    各モデルは全データから復元抽出した 80% で学習し、抽出されなかった標本を
    そのモデルの検証データとする（out-of-bag）。

    Returns
    -------
    (models, oob_scores)
        ``oob_scores`` は各モデルの out-of-bag R^2（6 ガス分）を並べた配列。
    """
    import keras
    from sklearn.metrics import r2_score
    from sklearn.utils import resample

    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    models, oob_scores = [], []

    for i in range(n_models):
        indices = np.arange(len(X))
        train_index = resample(indices, replace=True,
                               n_samples=round(len(X) * BOOTSTRAP_FRACTION),
                               random_state=random_state + i)
        oob_index = np.setdiff1d(indices, train_index)

        model = build_dnn(X.shape[1], seed=random_state + i)
        model.compile(loss=nan_mean_squared_error, optimizer=keras.optimizers.Adam())
        model.fit(X[train_index], y[train_index], epochs=epochs, batch_size=BATCH_SIZE,
                  validation_data=(X[oob_index], y[oob_index]), verbose=0)

        predicted = model.predict(X[oob_index], verbose=0)
        scores = np.array([r2_score(y[oob_index][:, k], predicted[:, k]) for k in range(y.shape[1])])
        models.append(model)
        oob_scores.append(scores)
        if verbose:
            print(f"model {i:2d}  out-of-bag R2 = " + "  ".join(f"{s:.3f}" for s in scores))

    return models, np.array(oob_scores)


def train_random_forest(X, y, random_state: int = RANDOM_STATE):
    """論文と同じ設定のマルチタスク・ランダムフォレストを学習する。

    論文 Methods:「200 estimators, ... square root of the number of features ...,
    and a max tree depth of 10」。
    """
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=200, max_depth=10, bootstrap=True,
                                  max_features="sqrt", random_state=random_state, n_jobs=-1)
    model.fit(np.asarray(X, dtype="float64"), np.asarray(y, dtype="float64"))
    return model


# -------------------------------------------------------------- アンサンブル推論
def ensemble_predict(models, X, batch_size: int = 4096):
    """アンサンブルの平均予測と、モデル間のばらつき（分散）を返す。

    分散は予測の不確かさの指標になる（論文「uncertainty quantification」）。
    """
    X = np.asarray(X, dtype="float64")
    predictions = np.array([m.predict(X, batch_size=batch_size, verbose=0) for m in models])
    return predictions.mean(axis=0), predictions.var(axis=0)


def score_per_gas(y_true, y_pred) -> "pd.DataFrame":
    """ガスごとの R^2・RMSE・MAE をまとめた表を返す。"""
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for k, gas in enumerate(GAS_NAMES):
        rows.append(
            {
                "gas": gas,
                "R2": r2_score(y_true[:, k], y_pred[:, k]),
                "RMSE": float(np.sqrt(mean_squared_error(y_true[:, k], y_pred[:, k]))),
                "MAE": mean_absolute_error(y_true[:, k], y_pred[:, k]),
            }
        )
    return pd.DataFrame(rows)
