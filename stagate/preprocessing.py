"""
データ前処理モジュール
論文に記載された前処理手順を厳密に実装
"""

import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Optional


def preprocess_data(
    adata: AnnData,
    n_top_genes: int = 3000,
    target_sum: float = 1e4,
    log_transform: bool = True,
    copy: bool = False
) -> Optional[AnnData]:
    """
    STAGATE 論文に基づくデータ前処理
    
    処理手順（論文 Methods セクションに記載）:
    1. 正規化: 各細胞の合計カウントを target_sum（デフォルト10,000）にスケール
    2. log1p 変換: log(x + 1)
    3. HVG（Highly Variable Genes）選択: デフォルト 3,000 遺伝子
    
    Parameters
    ----------
    adata : AnnData
        入力データ（空間トランスクリプトームデータ）
    n_top_genes : int, default=3000
        選択する HVG の数（論文では 3,000）
    target_sum : float, default=1e4
        正規化のターゲット合計値（論文では 10,000）
    log_transform : bool, default=True
        log1p 変換を実行するかどうか
    copy : bool, default=False
        コピーを返すかどうか
    
    Returns
    -------
    AnnData or None
        前処理済みの AnnData オブジェクト（copy=True の場合）
        copy=False の場合は None（元のオブジェクトを直接変更）
    
    Notes
    -----
    論文の記載:
    - "We normalized the gene expression matrix by scaling the total counts 
       of each spot to 10,000 followed by log-transformation."
    - "We selected 3,000 highly variable genes (HVGs) for downstream analysis."
    """
    if copy:
        adata = adata.copy()
    
    print("STAGATE データ前処理開始...")
    print(f"元のデータサイズ: {adata.shape}")
    
    # 生データを保存（後で必要になる場合のため）
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    
    # ステップ1: 正規化
    # 各細胞（スポット）の合計カウントを target_sum にスケール
    print(f"ステップ1: 正規化（target_sum={target_sum}）")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # ステップ2: log1p 変換
    if log_transform:
        print("ステップ2: log1p 変換")
        sc.pp.log1p(adata)
    
    # ステップ3: HVG 選択
    print(f"ステップ3: HVG 選択（n_top_genes={n_top_genes}）")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',  # 論文では Seurat 方式を使用
        subset=False  # 最初は subset せず、情報だけ保存
    )
    
    # HVG のみを抽出
    adata = adata[:, adata.var['highly_variable']].copy()
    
    print(f"前処理完了! 最終データサイズ: {adata.shape}")
    print(f"  - スポット数: {adata.n_obs}")
    print(f"  - HVG 数: {adata.n_vars}")
    
    if copy:
        return adata
    else:
        return None


def calculate_spatial_statistics(adata: AnnData, spatial_key: str = 'spatial') -> dict:
    """
    空間座標の統計情報を計算
    SNN 構築のパラメータ決定に使用
    
    Parameters
    ----------
    adata : AnnData
        空間座標を含む AnnData オブジェクト
    spatial_key : str, default='spatial'
        空間座標のキー（adata.obsm[spatial_key]）
    
    Returns
    -------
    dict
        統計情報（平均距離、中央値距離など）
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"spatial_key '{spatial_key}' not found in adata.obsm")
    
    coords = adata.obsm[spatial_key]
    
    # 最近傍距離を計算（サンプリング）
    from sklearn.neighbors import NearestNeighbors
    n_samples = min(1000, coords.shape[0])
    indices = np.random.choice(coords.shape[0], n_samples, replace=False)
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
    distances, _ = nbrs.kneighbors(coords[indices])
    nearest_distances = distances[:, 1]  # 自分自身を除く
    
    stats = {
        'mean_nearest_distance': np.mean(nearest_distances),
        'median_nearest_distance': np.median(nearest_distances),
        'std_nearest_distance': np.std(nearest_distances),
        'n_spots': coords.shape[0],
        'spatial_extent_x': coords[:, 0].max() - coords[:, 0].min(),
        'spatial_extent_y': coords[:, 1].max() - coords[:, 1].min(),
    }
    
    print("\n空間統計情報:")
    print(f"  - スポット数: {stats['n_spots']}")
    print(f"  - 平均最近傍距離: {stats['mean_nearest_distance']:.2f}")
    print(f"  - 中央値最近傍距離: {stats['median_nearest_distance']:.2f}")
    print(f"  - 空間範囲 (x): {stats['spatial_extent_x']:.2f}")
    print(f"  - 空間範囲 (y): {stats['spatial_extent_y']:.2f}")
    
    return stats


def filter_genes_by_expression(
    adata: AnnData,
    min_cells: int = 10,
    min_counts: int = 50,
    copy: bool = False
) -> Optional[AnnData]:
    """
    低発現遺伝子をフィルタリング
    オプショナルな前処理ステップ
    
    Parameters
    ----------
    adata : AnnData
        入力データ
    min_cells : int, default=10
        遺伝子が発現している最小細胞数
    min_counts : int, default=50
        遺伝子の最小カウント数
    copy : bool, default=False
        コピーを返すかどうか
    
    Returns
    -------
    AnnData or None
        フィルタリング済みデータ
    """
    if copy:
        adata = adata.copy()
    
    print(f"遺伝子フィルタリング（min_cells={min_cells}, min_counts={min_counts}）")
    print(f"フィルタリング前: {adata.n_vars} 遺伝子")
    
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # カウント数でもフィルタ
    gene_counts = np.array(adata.X.sum(axis=0)).flatten()
    keep_genes = gene_counts >= min_counts
    adata = adata[:, keep_genes].copy()
    
    print(f"フィルタリング後: {adata.n_vars} 遺伝子")
    
    if copy:
        return adata
    else:
        return None
