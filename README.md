# 高分子ガス分離膜の機械学習 — セミナー教材（全5回）

Science Advances 掲載論文を、ARIMデータポータル会員向けのセミナー教材として
Jupyter Notebook 5冊に再構成したものです。

> Jason Yang, Lei Tao, Jinlong He, Jeffrey R. McCutcheon, Ying Li,
> **"Machine learning enables interpretable discovery of innovative polymers for gas separation membranes"**,
> *Science Advances* **8**, eabn9545 (2022).
> DOI: [10.1126/sciadv.abn9545](https://doi.org/10.1126/sciadv.abn9545) （CC BY-NC 4.0）

原著の公開コード・データ（[github.com/jsunn-y/PolymerGasMembraneML](https://github.com/jsunn-y/PolymerGasMembraneML)）を
`code/` に同梱し、それを読み解きながら再現していく構成です。

---

## 目次

論文 Fig.1 のワークフロー（5ステップ）に1対1で対応しています。

| 回 | ノートブック | 論文のステップ | 内容 | Colab |
| --- | --- | --- | --- | --- |
| 第1回 | [`01_introduction_and_data.ipynb`](01_introduction_and_data.ipynb) | Step 1 | 課題設定（透過係数・選択性・Robeson上限）、データセットの理解、欠損の実態 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-PolymerInformatics/Advanced_Tutorial_3/blob/main/01_introduction_and_data.ipynb) |
| 第2回 | [`02_chemical_representation.ipynb`](02_chemical_representation.ipynb) | Step 2 | RDKit分子記述子とMorganフィンガープリント（MFF）の計算、特徴量の再現性 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-PolymerInformatics/Advanced_Tutorial_3/blob/main/02_chemical_representation.ipynb) |
| 第3回 | [`03_imputation_and_models.ipynb`](03_imputation_and_models.ipynb) | Step 3 | 欠損値補完（MICE）、マルチタスク学習、ランダムフォレスト vs DNNアンサンブル、**データリーケージの検証** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-PolymerInformatics/Advanced_Tutorial_3/blob/main/03_imputation_and_models.ipynb) |
| 第4回 | [`04_shap_interpretation.ipynb`](04_shap_interpretation.ipynb) | Step 3.5 | SHAPによるモデル解釈、化学的な設計指針の抽出、透過性/選択性トレードオフ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-PolymerInformatics/Advanced_Tutorial_3/blob/main/04_shap_interpretation.ipynb) |
| 第5回 | [`05_screening_and_discovery.ipynb`](05_screening_and_discovery.ipynb) | Step 4-5 | 大規模スクリーニング、候補材料の発見、外挿の不確かさ、シリーズ全体のまとめ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-Academy-PolymerInformatics/Advanced_Tutorial_3/blob/main/05_screening_and_discovery.ipynb) |

各回は独立して実行できますが、第1回から順に進めることを想定しています。

---

## 対象読者と前提知識

- ARIMデータポータル会員の研究者・技術者
- Pythonの基礎文法（変数・関数・for文）は理解している
- **高分子化学・膜分離の予備知識は不要**
- **統計学・機械学習の予備知識も不要**（必要な概念は本文で説明します）
- RDKit・scikit-learn・TensorFlow(Keras)・SHAP は初めてでも構いません

---

## 実行方法

### Google Colab（推奨）

上の表の Colab バッジをクリックするだけで実行できます。
各ノートブックの冒頭にセットアップセルがあり、リポジトリの取得と
必要なパッケージのインストールを自動で行います。

### ローカル環境

```bash
git clone https://github.com/ARIM-Academy-PolymerInformatics/Advanced_Tutorial_3.git
cd Advanced_Tutorial_3
pip install numpy pandas matplotlib seaborn scikit-learn rdkit shap tensorflow jupyter
jupyter lab
```

動作確認済みの環境：Python 3.10 / numpy 2.2 / pandas 2.3 / scikit-learn 1.7 /
rdkit 2026.03 / shap 0.49 / TensorFlow 2.21（Keras 3）

> 同梱の学習済みモデルは TensorFlow 2.3 時代の SavedModel 形式ですが、
> `module/pgm_models.py` がチェックポイントから重みを復元する方式を採っているため、
> 最新の TensorFlow でもそのまま利用できます（著者らが保存した予測値と
> 残差 1e-6 程度＝倍精度の丸め誤差の範囲で一致することを第3回で検証）。

---

## フォルダ構成

```
Advanced_Tutorial_3/
├── 01_introduction_and_data.ipynb      第1回
├── 02_chemical_representation.ipynb    第2回
├── 03_imputation_and_models.ipynb      第3回
├── 04_shap_interpretation.ipynb        第4回
├── 05_screening_and_discovery.ipynb    第5回
├── module/                             ヘルパー関数（下記参照）
├── output/                             ノートブックが生成する図・表・キャッシュ
├── code/                               原著リポジトリ（データ・学習済みモデル・元コード）
│   ├── datasets/                       Dataset A〜D
│   ├── pretrained_models/              著者らの学習済みDNNアンサンブル・候補リスト
│   ├── step2_generate_Xfeatures.py     原著コード（教材の元ネタ）
│   ├── step3_train.py
│   ├── step3.5_SHAP.py
│   ├── step4_screen.py
│   ├── DNN_functions.py
│   └── visualizations.ipynb
└── papers/                             原著論文PDF
```

### ヘルパーモジュール `module/`

**教材の本題ではないが必要な定型処理**は、ノートブック本文に展開せず
`module/` の `.py` に切り出しています。各ノートブックの冒頭で
「どのヘルパーを使うか」「原著コードから何を変えたか」を明示しています。

| ファイル | 役割 | 原著コードとの対応 |
| --- | --- | --- |
| `pgm_setup.py` | 実行環境の判定、パス解決、図の共通設定 | （新規） |
| `pgm_data.py` | データセットの読み込みと整形 | `step3_train.py` 冒頭の重複処理を集約 |
| `pgm_features.py` | RDKit記述子・Morganフィンガープリントの計算 | `step2_generate_Xfeatures.py` |
| `pgm_models.py` | DNNアンサンブル / RF の構築・学習・読み込み | `DNN_functions.py`, `step3_train.py` |
| `pgm_robeson.py` | Robesonプロットと上限線からの距離 | `visualizations.ipynb` の `plotRobeson()` |
| `pgm_shap.py` | SHAP値の計算・集計・可視化 | `step3.5_SHAP.py`, `plotSHAP()` |

原著コードからの変更は、いずれも**アルゴリズムではなく環境互換性・可読性**に関するものです。

- 非公開API（`tensorflow.python.ops.math_ops` 等）の排除。TF 2.6以降で動かないため
- 非推奨API（`AllChem.GetMorganFingerprint`）を `rdFingerprintGenerator` に置換
- `O(N²)` の線形探索を辞書引き（`O(1)`）に置換
- `os.chdir` による作業ディレクトリ書き換えの廃止
- 学習済みモデルの読み込み方式の変更（Keras 3 対応）

---

## 本教材の特徴：再現するだけでなく検証する

原著の結論をなぞるだけでなく、**公開コードを読んで手続きの妥当性を検証する**ことを
重視しています。第5回のまとめに検証結果の一覧表があります。主なものは次のとおりです。

**再現できたこと**

- 論文 Table 2（4モデルの R²）を小数点2桁まで再現
- `datasetAX_fing.csv`（Morganフィンガープリント）をSMILESからビット単位で完全再現
- 論文 Fig.3 の最重要記述子 `VSA_EState8` を再現
- 論文が化学的に議論した9つの部分構造すべてについて、効果の符号を再現
- 論文 Table 3 の「Robeson上限超えの候補数」を4つの分離すべてで再現

**検証の結果、注意が必要と分かったこと**

- 原著 `step3_train.py` は標準化を訓練/テスト分割の**前**に全データで行っている
  （影響はテスト R² で0.01程度と軽微）
- 原著の DNN アンサンブルは、ブートストラップ抽出を訓練データではなく**全データ**から行っており、
  **テストデータの全71件が平均9個のモデルの学習に混入**している。
  正しい手順で学習し直すと、テスト R² は 0.89 → 0.69 に下がる
- 欠損値補完は生データ778行に対して行われ、その後SMILESごとに平均されているため、
  学習ターゲットは実測値と補完値の混合になっている（最大3.8桁の食い違いが生じる例がある）
- 論文 Table 3 の「超高透過性の候補数（197件／225件）」は、
  著者らが公開した候補リストからは再現できない（桁違いに多く含まれる）
- RDKit のバージョン差により、`datasetAX_desc.csv` の列は記述子名が不明になっている
  （値照合で143/146列を特定）。Dataset D の MFF も11分子で微差がある

これらは原著の価値を否定するものではありません。
**著者らがコードとデータを公開してくれたからこそ検証できた**ものであり、
オープンサイエンスの意義を示す事例として教材に組み込んでいます。

---

## ライセンス・出典

- **論文本文・図**：CC BY-NC 4.0（Yang et al., *Sci. Adv.* 8, eabn9545, 2022）
- **原著コード・データ**：`code/LICENSE` を参照
- **データセットの出典**
  - Dataset A：PoLyInfo データベース、Membrane Society of Australasia (MSA) データベース
  - Dataset B：PI1M（Ma & Luo, *J. Chem. Inf. Model.* 60, 4684, 2020）
  - Dataset C：PubChem 由来のジアミン/ジイソシアネート × 酸二無水物の組み合わせ
  - Dataset D：既存ラダーポリマーの部品の組み合わせ + RNN
  - 欠損値補完：Yuan et al., *J. Membr. Sci.* 627, 119207 (2021)
- **本教材（ノートブックと `module/`）**：ARIMデータポータル セミナー教材

論文の図表を直接転載してはいません。図はすべて同梱データから再計算・再描画したものです。
