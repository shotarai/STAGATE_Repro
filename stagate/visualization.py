"""
可視化モジュール
論文 Figure 2-7 の再現可視化を実装
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from typing import Optional, Tuple, List
import scanpy as sc


def plot_spatial_domains(
    adata: AnnData,
    domain_key: str = 'domain',
    spatial_key: str = 'spatial',
    spot_size: float = 1.0,
    palette: Optional[str] = 'tab20',
    legend_loc: str = 'right margin',
    figsize: Tuple[int, int] = (10, 10),
    title: Optional[str] = None,
    save: Optional[str] = None,
    **kwargs
):
    """
    Spatial domain を空間座標上にプロット（論文 Fig.2, 3 相当）
    
    Parameters
    ----------
    adata : AnnData
        データ
    domain_key : str, default='domain'
        Domain ラベルのキー
    spatial_key : str, default='spatial'
        空間座標のキー
    spot_size : float, default=1.0
        スポットのサイズ
    palette : str, optional
        カラーパレット
    legend_loc : str, default='right margin'
        凡例の位置
    figsize : tuple, default=(10, 10)
        図のサイズ
    title : str, optional
        タイトル
    save : str, optional
        保存先のパス
    """
    if domain_key not in adata.obs:
        raise ValueError(f"'{domain_key}' not found in adata.obs")
    
    # Scanpy の spatial plot を使用
    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color=domain_key,
        size=spot_size,
        palette=palette,
        legend_loc=legend_loc,
        frameon=False,
        title=title if title else f'Spatial Domains ({domain_key})',
        show=False,
        **kwargs
    )
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def plot_umap(
    adata: AnnData,
    color: Optional[List[str]] = None,
    use_rep: str = 'STAGATE',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    figsize: Tuple[int, int] = (12, 5),
    save: Optional[str] = None,
    **kwargs
):
    """
    UMAP による低次元可視化（論文 Fig.4, 5 相当）
    
    Parameters
    ----------
    adata : AnnData
        データ
    color : list of str, optional
        色付けする変数（adata.obs のカラム名）
        指定しない場合は ['domain'] を使用
    use_rep : str, default='STAGATE'
        UMAP に使用する embeddings
    n_neighbors : int, default=15
        UMAP の n_neighbors パラメータ
    min_dist : float, default=0.1
        UMAP の min_dist パラメータ
    figsize : tuple, default=(12, 5)
        図のサイズ
    save : str, optional
        保存先のパス
    """
    if use_rep not in adata.obsm:
        raise ValueError(f"'{use_rep}' not found in adata.obsm")
    
    # UMAP を計算（まだ計算されていない場合）
    if 'X_umap' not in adata.obsm:
        print(f"UMAP を計算中（use_rep={use_rep}）...")
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        sc.tl.umap(adata, min_dist=min_dist)
    
    # デフォルトの色付け
    if color is None:
        color = ['domain'] if 'domain' in adata.obs else []
    
    # プロット
    sc.pl.umap(
        adata,
        color=color,
        frameon=False,
        show=False,
        **kwargs
    )
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def plot_attention_weights(
    adata: AnnData,
    edge_index: np.ndarray,
    attention_weights: np.ndarray,
    spatial_key: str = 'spatial',
    top_k: int = 1000,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'Reds',
    title: Optional[str] = None,
    save: Optional[str] = None
):
    """
    Attention weights を可視化（論文 Fig.5d-f 相当）
    
    Parameters
    ----------
    adata : AnnData
        データ
    edge_index : np.ndarray, shape (2, num_edges)
        エッジのインデックス
    attention_weights : np.ndarray, shape (num_edges,)
        各エッジの attention weight
    spatial_key : str, default='spatial'
        空間座標のキー
    top_k : int, default=1000
        描画する attention weight の高いエッジの数
    figsize : tuple, default=(12, 10)
        図のサイズ
    cmap : str, default='Reds'
        カラーマップ
    title : str, optional
        タイトル
    save : str, optional
        保存先のパス
    """
    coords = adata.obsm[spatial_key]
    
    # Top-k のエッジを選択
    if attention_weights.ndim > 1:
        # Multi-head の場合は平均を取る
        attention_weights = attention_weights.mean(axis=1)
    
    top_indices = np.argsort(attention_weights)[-top_k:]
    edge_index_top = edge_index[:, top_indices]
    weights_top = attention_weights[top_indices]
    
    # 正規化
    weights_norm = (weights_top - weights_top.min()) / (weights_top.max() - weights_top.min())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # エッジを描画（attention weight で色付け）
    for i, (src, dst) in enumerate(edge_index_top.T):
        x = [coords[src, 0], coords[dst, 0]]
        y = [coords[src, 1], coords[dst, 1]]
        color = plt.cm.get_cmap(cmap)(weights_norm[i])
        ax.plot(x, y, color=color, alpha=0.6, linewidth=1.5)
    
    # スポットを描画
    ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=10, zorder=2, alpha=0.5)
    
    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=weights_top.min(), vmax=weights_top.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title if title else f'Attention Weights (Top-{top_k} edges)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def plot_gene_expression_spatial(
    adata: AnnData,
    genes: List[str],
    spatial_key: str = 'spatial',
    layer: Optional[str] = None,
    ncols: int = 3,
    spot_size: float = 1.0,
    cmap: str = 'viridis',
    figsize: Optional[Tuple[int, int]] = None,
    save: Optional[str] = None,
    **kwargs
):
    """
    遺伝子発現を空間座標上にプロット
    
    Parameters
    ----------
    adata : AnnData
        データ
    genes : list of str
        可視化する遺伝子名のリスト
    spatial_key : str, default='spatial'
        空間座標のキー
    layer : str, optional
        使用するレイヤー（指定しない場合は adata.X）
    ncols : int, default=3
        列数
    spot_size : float, default=1.0
        スポットのサイズ
    cmap : str, default='viridis'
        カラーマップ
    figsize : tuple, optional
        図のサイズ（自動計算される）
    save : str, optional
        保存先のパス
    """
    # 遺伝子が存在するかチェック
    genes_available = [g for g in genes if g in adata.var_names]
    if len(genes_available) == 0:
        raise ValueError("指定された遺伝子が見つかりません")
    
    if len(genes_available) < len(genes):
        missing = set(genes) - set(genes_available)
        print(f"警告: 以下の遺伝子が見つかりません: {missing}")
    
    # Scanpy の spatial plot を使用
    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color=genes_available,
        layer=layer,
        size=spot_size,
        cmap=cmap,
        ncols=ncols,
        frameon=False,
        show=False,
        **kwargs
    )
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def plot_domain_comparison(
    adata: AnnData,
    domain_keys: List[str],
    spatial_key: str = 'spatial',
    ncols: int = 3,
    spot_size: float = 1.0,
    palette: str = 'tab20',
    figsize: Optional[Tuple[int, int]] = None,
    save: Optional[str] = None
):
    """
    複数の domain 予測結果を比較（論文 Fig.6 相当）
    
    Parameters
    ----------
    adata : AnnData
        データ
    domain_keys : list of str
        比較する domain ラベルのキーのリスト
    spatial_key : str, default='spatial'
        空間座標のキー
    ncols : int, default=3
        列数
    spot_size : float, default=1.0
        スポットのサイズ
    palette : str, default='tab20'
        カラーパレット
    figsize : tuple, optional
        図のサイズ
    save : str, optional
        保存先のパス
    """
    # 利用可能な domain keys をチェック
    available_keys = [k for k in domain_keys if k in adata.obs]
    if len(available_keys) == 0:
        raise ValueError("指定された domain_keys が見つかりません")
    
    # Scanpy の spatial plot を使用
    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color=available_keys,
        size=spot_size,
        palette=palette,
        ncols=ncols,
        frameon=False,
        show=False
    )
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()


def plot_embeddings_heatmap(
    adata: AnnData,
    use_rep: str = 'STAGATE',
    cluster_key: str = 'domain',
    n_spots_sample: int = 1000,
    figsize: Tuple[int, int] = (12, 10),
    save: Optional[str] = None
):
    """
    Embeddings のヒートマップ（クラスタごとに並び替え）
    
    Parameters
    ----------
    adata : AnnData
        データ
    use_rep : str, default='STAGATE'
        使用する embeddings
    cluster_key : str, default='domain'
        クラスタラベル
    n_spots_sample : int, default=1000
        サンプリングするスポット数（多すぎる場合）
    figsize : tuple, default=(12, 10)
        図のサイズ
    save : str, optional
        保存先のパス
    """
    if use_rep not in adata.obsm:
        raise ValueError(f"'{use_rep}' not found in adata.obsm")
    
    if cluster_key not in adata.obs:
        raise ValueError(f"'{cluster_key}' not found in adata.obs")
    
    embeddings = adata.obsm[use_rep]
    clusters = adata.obs[cluster_key].values
    
    # サンプリング
    if adata.n_obs > n_spots_sample:
        indices = np.random.choice(adata.n_obs, n_spots_sample, replace=False)
        embeddings = embeddings[indices]
        clusters = clusters[indices]
    
    # クラスタごとにソート
    sort_idx = np.argsort(clusters)
    embeddings_sorted = embeddings[sort_idx]
    clusters_sorted = clusters[sort_idx]
    
    # プロット
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(
        embeddings_sorted.T,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    
    ax.set_xlabel('Spots (sorted by domain)')
    ax.set_ylabel('Latent dimensions')
    ax.set_title(f'STAGATE Embeddings Heatmap')
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Embedding value', rotation=270, labelpad=20)
    
    # クラスタの境界線を描画
    unique_clusters = np.unique(clusters_sorted)
    boundaries = []
    for i, cluster in enumerate(unique_clusters[:-1]):
        boundary = np.where(clusters_sorted == cluster)[0][-1]
        boundaries.append(boundary)
        ax.axvline(boundary, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()
