"""化学特徴量（RDKit 分子記述子・Morgan fingerprint with frequency）の計算。

原著リポジトリの `code/step2_generate_Xfeatures.py` を、教材で使いやすい形に
書き直したものです。アルゴリズムは原著と同一で、次の3点だけを変更しています。

1. 非推奨 API の置き換え
   `AllChem.GetMorganFingerprint` は RDKit 2022.09 以降で非推奨。
   本モジュールでは `rdFingerprintGenerator.GetMorganGenerator` を使う。
   両者は同じハッシュ値を返すため、原著の `datasetAX_fing.csv` を
   ビット単位で再現できることを確認済み（02冊目で検証する）。
2. `O(N^2)` の線形探索を辞書引き（`O(1)`）に置き換え、353 分子でも一瞬で終わるようにした。
   原著は部分構造ごとに `Corr_df[Corr_df[0] == key]` という DataFrame 全走査を行っていた。
3. `os.chdir` による作業ディレクトリの書き換えをやめ、引数でパスを受け取る形にした。

これらは「機械学習の理解」には寄与しない実装上の都合なので、ノートブック本文には
展開せずこのモジュールに置いています。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MORGAN_RADIUS = 3
"""Morgan fingerprint の半径。論文の Methods 節に従い 3（結合3つ分の環境まで見る）。"""

MAX_ZEROS = 325
"""部分構造を残す条件。353 分子のうちゼロの分子数がこの値未満なら採用する。

原著 `step2_generate_Xfeatures.py` の `NumberOfZero = 325` と同じ値。
353 - 325 = 28 分子以上に出現する部分構造だけを残すことになり、
結果として論文本文の「最頻出 114 部分構造」が得られる。
"""


# ---------------------------------------------------------------- 分子記述子
def smiles_to_mols(smiles_list, keep_invalid: bool = False):
    """SMILES 文字列のリストを RDKit の Mol オブジェクトに変換する。

    Parameters
    ----------
    keep_invalid : bool
        True なら変換に失敗した分子を ``None`` のまま残す。False なら除外する。

    Returns
    -------
    (mols, valid_mask)
        ``valid_mask`` は元のリストと同じ長さの bool 配列。
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # パースエラーの大量出力を抑止
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_mask = np.array([m is not None for m in mols])
    if not keep_invalid:
        mols = [m for m in mols if m is not None]
    return mols, valid_mask


def compute_descriptors(mols, descriptor_names=None) -> pd.DataFrame:
    """RDKit の全記述子を計算し、記述子名を列名とする DataFrame を返す。

    論文が使った RDKit（2020年頃）の記述子は 208 個だったが、現在の RDKit では
    217 個に増えている。そのため列の「位置番号」は論文当時と一致しない。
    本モジュールは位置番号ではなく記述子名で扱うことでこの問題を回避する。
    """
    from rdkit.Chem import Descriptors

    desc_list = Descriptors.descList
    if descriptor_names is not None:
        wanted = set(descriptor_names)
        desc_list = [(n, f) for n, f in desc_list if n in wanted]

    names = [n for n, _ in desc_list]
    values = np.full((len(mols), len(desc_list)), np.nan)
    for i, mol in enumerate(mols):
        for j, (_, func) in enumerate(desc_list):
            try:
                values[i, j] = func(mol)
            except Exception:  # 一部の記述子は特定の分子で失敗しうる
                values[i, j] = np.nan
    return pd.DataFrame(values, columns=names)


def drop_uninformative_descriptors(frame: pd.DataFrame) -> pd.DataFrame:
    """欠損を含む列と、全行がゼロの列を落とす（原著 step2 と同じ前処理）。"""
    frame = frame.dropna(axis="columns")
    return frame.loc[:, (frame != 0).any(axis=0)]


def identify_legacy_descriptor_columns(reference: pd.DataFrame, computed: pd.DataFrame) -> pd.DataFrame:
    """位置番号だけが残っている `datasetAX_desc.csv` の列に記述子名を割り当てる。

    同梱の `datasetAX_desc.csv` は列名が当時の `Descriptors.descList` の位置番号
    （0〜207）になっており、そのままでは何の記述子か分からない。ここでは
    「現在の RDKit で計算した値と完全に一致する記述子」を探すことで名前を復元する。

    値が完全に一致する記述子が複数ある場合（例: この 353 分子では
    ``fr_Al_OH`` と ``fr_Al_OH_noTert`` が同じ値になる）は候補をすべて返す。
    """
    records = []
    computed_values = computed.to_numpy()
    for column in reference.columns:
        target = reference[column].to_numpy(dtype=float)
        matches = [
            computed.columns[j]
            for j in range(computed_values.shape[1])
            if np.allclose(target, computed_values[:, j], rtol=1e-6, atol=1e-8, equal_nan=True)
        ]
        records.append(
            {
                "legacy_index": int(column),
                "n_candidates": len(matches),
                "descriptor_name": matches[0] if matches else "(unmatched)",
                "all_candidates": ", ".join(matches) if matches else "",
            }
        )
    return pd.DataFrame(records).sort_values("legacy_index").reset_index(drop=True)


# -------------------------------------------- Morgan fingerprint with frequency
def build_mff_vocabulary(mols, radius: int = MORGAN_RADIUS, max_zeros: int = MAX_ZEROS):
    """学習セット（Dataset A）から MFF の「語彙」を構築する。

    Returns
    -------
    dict
        ``hash_codes``   : 出現した全部分構造のハッシュ値のリスト（順序が語彙の索引になる）
        ``selected_index``: 採用した部分構造の索引（= 教材で使う列名）
        ``selected_hash`` : 採用した部分構造のハッシュ値
        ``counts``        : Dataset A における全部分構造の出現行列（DataFrame）
    """
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    per_molecule = [generator.GetSparseCountFingerprint(m).GetNonzeroElements() for m in mols]

    hash_codes = []
    for counts in per_molecule:
        hash_codes.extend(counts.keys())
    vocabulary = list(set(hash_codes))
    position = {code: i for i, code in enumerate(vocabulary)}

    matrix = np.zeros((len(mols), len(vocabulary)), dtype=np.int32)
    for row, counts in enumerate(per_molecule):
        for code, frequency in counts.items():
            matrix[row, position[code]] = frequency
    counts_frame = pd.DataFrame(matrix)

    n_zeros = (counts_frame == 0).sum(axis=0)
    selected_index = n_zeros[n_zeros < max_zeros].index.to_numpy()
    return {
        "hash_codes": vocabulary,
        "selected_index": selected_index,
        "selected_hash": np.array([vocabulary[i] for i in selected_index]),
        "counts": counts_frame,
    }


def compute_mff(mols, vocabulary: dict, radius: int = MORGAN_RADIUS) -> pd.DataFrame:
    """既存の語彙を使って、任意の分子集合の MFF 特徴量を計算する。

    スクリーニング対象（Dataset B/C/D）は Dataset A で作った語彙で表現しなければ
    ならない。学習時と推論時で特徴量の意味が揃っている必要があるため。
    """
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    selected_hash = vocabulary["selected_hash"]
    column_of = {code: j for j, code in enumerate(selected_hash)}

    matrix = np.zeros((len(mols), len(selected_hash)), dtype=np.int32)
    for row, mol in enumerate(mols):
        counts = generator.GetSparseCountFingerprint(mol).GetNonzeroElements()
        for code, frequency in counts.items():
            column = column_of.get(code)
            if column is not None:
                matrix[row, column] = frequency
    return pd.DataFrame(matrix, columns=[str(i) for i in vocabulary["selected_index"]])


def draw_substructure(mol, hash_code: int, radius: int = MORGAN_RADIUS, size=(220, 200)):
    """指定したハッシュ値の部分構造が分子中のどこに当たるかを描画する。

    論文 Fig.4C に相当する図を作るための関数。該当する部分構造が分子中に
    見つからない場合は ``None`` を返す。
    """
    from rdkit.Chem import Draw
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    info = rdFingerprintGenerator.AdditionalOutput()
    info.AllocateBitInfoMap()
    generator.GetSparseCountFingerprint(mol, additionalOutput=info)
    bit_info = info.GetBitInfoMap()
    if hash_code not in bit_info:
        return None
    drawing = Draw.DrawMorganBit(mol, hash_code, bit_info, molSize=size, useSVG=True)
    # RDKit は IPython 環境では SVG 表示オブジェクトを、それ以外では文字列を返す。
    # 呼び出し側を単純にするため、常に SVG 文字列に揃える。
    return drawing if isinstance(drawing, str) else drawing.data


def find_molecule_with_substructure(mols, hash_code: int, radius: int = MORGAN_RADIUS):
    """指定した部分構造を含む分子を先頭から1つ探して返す。見つからなければ ``None``。"""
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    for mol in mols:
        if hash_code in generator.GetSparseCountFingerprint(mol).GetNonzeroElements():
            return mol
    return None
