"""
Cell type-aware SNN 構築モジュール
論文 Fig.1c および Methods の "cell type-aware SNN" を実装
"""

import numpy as np
import torch
import scanpy as sc
from anndata import AnnData
from typing import Optional, Tuple
from scipy.sparse import csr_matrix
import warnings


def build_celltype_aware_snn(
    adata: AnnData,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    louvain_resolution: float = 0.2,
    use_rep: str = 'X',
    n_neighbors: int = 15,
    copy: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cell type-aware SNN を構築
    
    論文の記載:
    "To construct a more biologically meaningful SNN, we performed a 
     pre-clustering step using the Louvain algorithm with resolution 
     parameter of 0.2, and then removed edges connecting spots from 
     different clusters (cell type-aware SNN)."
    
    手順:
    1. Louvain アルゴリズムで pre-clustering（resolution=0.2）
    2. 異なるクラスタに属するスポット間のエッジを除去
    3. pruning された SNN を返す
    
    Parameters
    ----------
    adata : AnnData
        遺伝子発現データを含む AnnData オブジェクト
    edge_index : torch.Tensor, shape (2, num_edges)
        元の SNN のエッジインデックス
    edge_attr : torch.Tensor, shape (num_edges,)
        元の SNN のエッジ重み
    louvain_resolution : float, default=0.2
        Louvain アルゴリズムの resolution パラメータ（論文では 0.2）
    use_rep : str, default='X'
        クラスタリングに使用する特徴量
        'X': 正規化済み遺伝子発現
        'X_pca': PCA 後の特徴量
    n_neighbors : int, default=15
        近傍グラフ構築時の近傍数
    copy : bool, default=False
        adata をコピーするかどうか
    
    Returns
    -------
    pruned_edge_index : torch.Tensor, shape (2, num_pruned_edges)
        Pruning 後のエッジインデックス
    pruned_edge_attr : torch.Tensor, shape (num_pruned_edges,)
        Pruning 後のエッジ重み
    
    Notes
    -----
    論文での効果:
    - cell type-aware SNN は、より生物学的に意味のある spatial domain を検出
    - 異なる細胞タイプの境界をより明確に分離
    - ablation study で性能向上を確認
    """
    if copy:
        adata = adata.copy()
    
    print(f"\n{'='*60}")
    print("Cell type-aware SNN 構築")
    print(f"{'='*60}")
    print(f"元のエッジ数: {edge_index.shape[1]}")
    
    # ステップ1: PCA（必要な場合）
    if use_rep == 'X_pca' and 'X_pca' not in adata.obsm:
        print("PCA を実行中...")
        sc.tl.pca(adata, n_comps=50)
    
    # ステップ2: 近傍グラフ構築
    print(f"近傍グラフ構築（n_neighbors={n_neighbors}）...")
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=0
    )
    
    # ステップ3: Louvain pre-clustering
    print(f"Louvain pre-clustering（resolution={louvain_resolution}）...")
    sc.tl.louvain(
        adata,
        resolution=louvain_resolution,
        key_added='louvain_preclustering',
        random_state=0
    )
    
    n_clusters = len(adata.obs['louvain_preclustering'].unique())
    print(f"検出されたクラスタ数: {n_clusters}")
    
    # クラスタサイズの分布を表示
    cluster_sizes = adata.obs['louvain_preclustering'].value_counts().sort_index()
    print("\nクラスタサイズ分布:")
    for cluster_id, size in cluster_sizes.items():
        print(f"  クラスタ {cluster_id}: {size} スポット")
    
    # ステップ4: 異なるクラスタ間のエッジを除去
    print("\nエッジの pruning を実行中...")
    cluster_labels = adata.obs['louvain_preclustering'].values
    
    # エッジごとにクラスタが同じかチェック
    src_clusters = cluster_labels[edge_index[0].numpy()]
    dst_clusters = cluster_labels[edge_index[1].numpy()]
    same_cluster_mask = src_clusters == dst_clusters
    
    # Pruning 後のエッジ
    pruned_edge_index = edge_index[:, same_cluster_mask]
    pruned_edge_attr = edge_attr[same_cluster_mask]
    
    n_removed = edge_index.shape[1] - pruned_edge_index.shape[1]
    removal_rate = n_removed / edge_index.shape[1] * 100
    
    print(f"\n結果:")
    print(f"  - 除去されたエッジ数: {n_removed}")
    print(f"  - 除去率: {removal_rate:.2f}%")
    print(f"  - 残存エッジ数: {pruned_edge_index.shape[1]}")
    print(f"  - 平均次数（元）: {edge_index.shape[1] / adata.n_obs:.2f}")
    print(f"  - 平均次数（後）: {pruned_edge_index.shape[1] / adata.n_obs:.2f}")
    print(f"{'='*60}\n")
    
    # 警告チェック
    avg_degree_after = pruned_edge_index.shape[1] / adata.n_obs
    if avg_degree_after < 2:
        warnings.warn(
            f"Pruning 後の平均次数が低すぎます ({avg_degree_after:.2f} < 2). "
            f"louvain_resolution を小さくするか、元の SNN の radius/k を大きくしてください。",
            UserWarning
        )
    
    # AnnData に保存
    adata.uns['celltype_aware_snn'] = {
        'louvain_resolution': louvain_resolution,
        'n_clusters': n_clusters,
        'n_removed_edges': n_removed,
        'removal_rate': removal_rate,
        'edge_index': pruned_edge_index.numpy(),
        'edge_attr': pruned_edge_attr.numpy()
    }
    
    return pruned_edge_index, pruned_edge_attr


def visualize_preclustering(
    adata: AnnData,
    spatial_key: str = 'spatial',
    cluster_key: str = 'louvain_preclustering',
    figsize: Tuple[int, int] = (12, 5),
    save: Optional[str] = None
):
    """
    Pre-clustering の結果を可視化
    
    Parameters
    ----------
    adata : AnnData
        クラスタリング結果を含む AnnData オブジェクト
    spatial_key : str, default='spatial'
        空間座標のキー
    cluster_key : str, default='louvain_preclustering'
        クラスタラベルのキー
    figsize : tuple, default=(12, 5)
        図のサイズ
    save : str, optional
        保存先のパス
    """
    import matplotlib.pyplot as plt
    
    if cluster_key not in adata.obs:
        raise ValueError(f"cluster_key '{cluster_key}' not found in adata.obs")
    
    coords = adata.obsm[spatial_key]
    clusters = adata.obs[cluster_key]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左: 空間プロット
    ax = axes[0]
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=clusters.astype('category').cat.codes,
        cmap='tab20',
        s=20
    )
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Pre-clustering (spatial)\n{len(clusters.unique())} clusters')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax)
    
    # 右: クラスタサイズの分布
    ax = axes[1]
    cluster_counts = clusters.value_counts().sort_index()
    ax.bar(range(len(cluster_counts)), cluster_counts.values)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of spots')
    ax.set_title('Cluster size distribution')
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels(cluster_counts.index)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def compare_snn_pruning(
    adata: AnnData,
    edge_index_original: torch.Tensor,
    edge_index_pruned: torch.Tensor,
    spatial_key: str = 'spatial',
    max_edges: int = 3000,
    figsize: Tuple[int, int] = (16, 7),
    save: Optional[str] = None
):
    """
    元の SNN と pruning 後の SNN を比較可視化
    
    Parameters
    ----------
    adata : AnnData
        空間座標を含む AnnData オブジェクト
    edge_index_original : torch.Tensor
        元のエッジインデックス
    edge_index_pruned : torch.Tensor
        Pruning 後のエッジインデックス
    spatial_key : str, default='spatial'
        空間座標のキー
    max_edges : int, default=3000
        描画する最大エッジ数
    figsize : tuple, default=(16, 7)
        図のサイズ
    save : str, optional
        保存先のパス
    """
    import matplotlib.pyplot as plt
    
    coords = adata.obsm[spatial_key]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 左: 元の SNN
    ax = axes[0]
    _plot_network(ax, coords, edge_index_original, max_edges, 'Original SNN')
    
    # 右: Pruning 後の SNN
    ax = axes[1]
    _plot_network(ax, coords, edge_index_pruned, max_edges, 'Cell type-aware SNN (pruned)')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def _plot_network(ax, coords, edge_index, max_edges, title):
    """ネットワークをプロットするヘルパー関数"""
    n_edges = edge_index.shape[1]
    
    # サンプリング
    if n_edges > max_edges:
        indices = np.random.choice(n_edges, max_edges, replace=False)
        edge_index_sampled = edge_index[:, indices]
    else:
        edge_index_sampled = edge_index
    
    # エッジを描画
    for i in range(edge_index_sampled.shape[1]):
        src, dst = edge_index_sampled[:, i]
        x = [coords[src, 0], coords[dst, 0]]
        y = [coords[src, 1], coords[dst, 1]]
        ax.plot(x, y, 'gray', alpha=0.2, linewidth=0.5)
    
    # スポットを描画
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=10, zorder=2)
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'{title}\n({n_edges} edges)')
    ax.set_aspect('equal')
