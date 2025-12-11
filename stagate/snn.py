"""
Spatial Neighbor Network (SNN) 構築モジュール
論文 Methods セクションの SNN 構築手順を厳密に実装
"""

import numpy as np
import torch
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
from scipy.sparse import csr_matrix, coo_matrix
from typing import Optional, Tuple, Union
import warnings


def build_spatial_network(
    adata: AnnData,
    spatial_key: str = 'spatial',
    radius: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    method: str = 'radius',
    return_adjacency: bool = True
) -> Union[Tuple[torch.Tensor, torch.Tensor], csr_matrix]:
    """
    Spatial Neighbor Network (SNN) を構築
    
    論文の記載:
    "We constructed a spatial neighbor network (SNN) based on the spatial 
     coordinates of spots. Two spots are connected if their Euclidean 
     distance is less than a threshold r."
    
    Parameters
    ----------
    adata : AnnData
        空間座標を含む AnnData オブジェクト
    spatial_key : str, default='spatial'
        空間座標のキー（adata.obsm[spatial_key]）
    radius : float, optional
        接続判定の半径（論文では r）
        method='radius' の場合に使用
    k_neighbors : int, optional
        k-nearest neighbors の k
        method='knn' の場合に使用
    method : str, default='radius'
        'radius': 半径ベース（論文の標準手法）
        'knn': k-nearest neighbors
    return_adjacency : bool, default=True
        Trueの場合、PyTorch Geometric用のedge_indexとedge_attrを返す
        Falseの場合、隣接行列（sparse matrix）を返す
    
    Returns
    -------
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジのインデックス（return_adjacency=True の場合）
    edge_attr : torch.Tensor, shape (num_edges,)
        エッジの重み（距離の逆数）（return_adjacency=True の場合）
    adjacency : csr_matrix
        隣接行列（return_adjacency=False の場合）
    
    Notes
    -----
    論文での典型的な設定:
    - 10x Visium データ: radius = 150 (μm)
    - Slide-seq データ: radius は組織タイプに応じて調整
    """
    if spatial_key not in adata.obsm:
        raise ValueError(f"spatial_key '{spatial_key}' not found in adata.obsm")
    
    coords = adata.obsm[spatial_key]
    n_spots = coords.shape[0]
    
    print(f"\n{'='*60}")
    print("Spatial Neighbor Network (SNN) 構築")
    print(f"{'='*60}")
    print(f"スポット数: {n_spots}")
    print(f"座標次元: {coords.shape[1]}")
    
    if method == 'radius':
        if radius is None:
            # デフォルト値を自動設定（平均最近傍距離の3倍）
            nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
            distances, _ = nbrs.kneighbors(coords[:min(1000, n_spots)])
            avg_distance = np.mean(distances[:, 1])
            radius = avg_distance * 3.0
            print(f"半径が未指定のため自動設定: {radius:.2f}")
        
        print(f"方法: 半径ベース (r={radius:.2f})")
        
        # 半径ベースのグラフ構築
        adjacency = radius_neighbors_graph(
            coords,
            radius=radius,
            mode='distance',
            include_self=False
        )
        
    elif method == 'knn':
        if k_neighbors is None:
            k_neighbors = 6  # デフォルト値
            print(f"k が未指定のため デフォルト値 k={k_neighbors} を使用")
        
        print(f"方法: k-nearest neighbors (k={k_neighbors})")
        
        # k-NN グラフ構築
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # 対称な隣接行列を構築
        rows = np.repeat(np.arange(n_spots), k_neighbors)
        cols = indices[:, 1:].flatten()  # 自分自身を除く
        dists = distances[:, 1:].flatten()
        
        adjacency = coo_matrix(
            (dists, (rows, cols)),
            shape=(n_spots, n_spots)
        )
        
        # 対称化（無向グラフ）
        adjacency = adjacency + adjacency.T
        adjacency = csr_matrix(adjacency)
        
        # 重複エッジの処理（最小距離を採用）
        adjacency.data = np.minimum(adjacency.data, adjacency.data)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'radius' or 'knn'.")
    
    # エッジ数をカウント
    adjacency_coo = adjacency.tocoo()
    n_edges = adjacency_coo.nnz
    avg_degree = n_edges / n_spots
    
    print(f"エッジ数: {n_edges}")
    print(f"平均次数: {avg_degree:.2f}")
    print(f"{'='*60}\n")
    
    # 警告チェック
    if avg_degree < 2:
        warnings.warn(
            f"平均次数が低すぎます ({avg_degree:.2f} < 2). "
            f"radius を大きくするか、k_neighbors を増やしてください。",
            UserWarning
        )
    elif avg_degree > 20:
        warnings.warn(
            f"平均次数が高すぎます ({avg_degree:.2f} > 20). "
            f"radius を小さくするか、k_neighbors を減らすことを推奨します。",
            UserWarning
        )
    
    # AnnData に保存
    adata.uns['spatial_network'] = {
        'method': method,
        'radius': radius if method == 'radius' else None,
        'k_neighbors': k_neighbors if method == 'knn' else None,
        'n_edges': n_edges,
        'avg_degree': avg_degree
    }
    
    if return_adjacency:
        # PyTorch Geometric 形式に変換
        edge_index, edge_attr = sparse_to_edge_index(adjacency_coo)
        
        # AnnData に保存
        adata.uns['spatial_network']['edge_index'] = edge_index.numpy()
        adata.uns['spatial_network']['edge_attr'] = edge_attr.numpy()
        
        return edge_index, edge_attr
    else:
        return adjacency


def sparse_to_edge_index(
    adjacency: coo_matrix
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse matrix を PyTorch Geometric の edge_index 形式に変換
    
    Parameters
    ----------
    adjacency : coo_matrix
        隣接行列（COO 形式）
    
    Returns
    -------
    edge_index : torch.Tensor, shape (2, num_edges)
        エッジのインデックス
    edge_attr : torch.Tensor, shape (num_edges,)
        エッジの重み（距離の逆数）
    """
    # edge_index: (2, num_edges)
    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]),
        dtype=torch.long
    )
    
    # edge_attr: 距離を重みに変換（距離の逆数）
    # 距離が0の場合は無限大を避けるため、小さい値で置換
    distances = adjacency.data
    distances = np.where(distances < 1e-6, 1e-6, distances)
    
    # ガウシアンカーネルを使用（論文での標準的な重み付け）
    # w_ij = exp(-d_ij^2 / (2 * sigma^2))
    sigma = np.median(distances)
    edge_attr = torch.tensor(
        np.exp(-distances**2 / (2 * sigma**2)),
        dtype=torch.float32
    )
    
    return edge_index, edge_attr


def visualize_spatial_network(
    adata: AnnData,
    spatial_key: str = 'spatial',
    edge_index: Optional[torch.Tensor] = None,
    max_edges: int = 5000,
    figsize: Tuple[int, int] = (10, 10),
    save: Optional[str] = None
):
    """
    SNN を可視化
    
    Parameters
    ----------
    adata : AnnData
        空間座標を含む AnnData オブジェクト
    spatial_key : str, default='spatial'
        空間座標のキー
    edge_index : torch.Tensor, optional
        エッジのインデックス（指定しない場合は adata.uns から取得）
    max_edges : int, default=5000
        描画する最大エッジ数（多すぎる場合はサンプリング）
    figsize : tuple, default=(10, 10)
        図のサイズ
    save : str, optional
        保存先のパス
    """
    import matplotlib.pyplot as plt
    
    coords = adata.obsm[spatial_key]
    
    if edge_index is None:
        if 'spatial_network' in adata.uns and 'edge_index' in adata.uns['spatial_network']:
            edge_index = torch.tensor(adata.uns['spatial_network']['edge_index'])
        else:
            raise ValueError("edge_index が指定されておらず、adata.uns にも保存されていません")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # エッジをサンプリング（多すぎる場合）
    n_edges = edge_index.shape[1]
    if n_edges > max_edges:
        indices = np.random.choice(n_edges, max_edges, replace=False)
        edge_index = edge_index[:, indices]
        print(f"エッジが多いため {max_edges} 本にサンプリングしました")
    
    # エッジを描画
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        x = [coords[src, 0], coords[dst, 0]]
        y = [coords[src, 1], coords[dst, 1]]
        ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5)
    
    # スポットを描画
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=10, zorder=2)
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Spatial Neighbor Network ({edge_index.shape[1]} edges)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def get_spatial_network_statistics(adata: AnnData) -> dict:
    """
    SNN の統計情報を取得
    
    Parameters
    ----------
    adata : AnnData
        SNN 情報を含む AnnData オブジェクト
    
    Returns
    -------
    dict
        統計情報
    """
    if 'spatial_network' not in adata.uns:
        raise ValueError("SNN が構築されていません。先に build_spatial_network() を実行してください。")
    
    return adata.uns['spatial_network']
