"""
クラスタリングモジュール
STAGATE embeddings を使った spatial domain のクラスタリング
"""

import numpy as np
import scanpy as sc
from anndata import AnnData
from typing import Optional, Literal
import warnings


def cluster_domains(
    adata: AnnData,
    use_rep: str = 'STAGATE',
    method: Literal['louvain', 'leiden', 'mclust'] = 'louvain',
    resolution: float = 0.5,
    n_clusters: Optional[int] = None,
    key_added: str = 'domain',
    n_neighbors: int = 15,
    random_state: int = 0
) -> AnnData:
    """
    STAGATE embeddings を使って spatial domain をクラスタリング
    
    論文 Methods:
    "We applied the Louvain algorithm or mclust to cluster the latent 
     representations into spatial domains."
    
    Parameters
    ----------
    adata : AnnData
        STAGATE embeddings を含む AnnData オブジェクト
        (adata.obsm[use_rep] に embeddings が保存されている)
    use_rep : str, default='STAGATE'
        使用する embeddings のキー
    method : {'louvain', 'leiden', 'mclust'}, default='louvain'
        クラスタリング手法
        - 'louvain': Louvain アルゴリズム（論文の標準）
        - 'leiden': Leiden アルゴリズム（Louvain の改良版）
        - 'mclust': Model-based clustering（R の mclust）
    resolution : float, default=0.5
        Louvain/Leiden の resolution パラメータ
        大きいほど多くのクラスタが生成される
    n_clusters : int, optional
        mclust 使用時のクラスタ数（指定しない場合は自動決定）
    key_added : str, default='domain'
        クラスタリング結果を保存するキー（adata.obs[key_added]）
    n_neighbors : int, default=15
        近傍グラフ構築時の近傍数
    random_state : int, default=0
        乱数シード
    
    Returns
    -------
    adata : AnnData
        クラスタリング結果が追加された AnnData オブジェクト
    
    Notes
    -----
    論文での推奨設定:
    - DLPFC データセット: Louvain, resolution ~ 0.4-0.6
    - Mouse brain データセット: Louvain, resolution ~ 0.5
    """
    if use_rep not in adata.obsm:
        raise ValueError(f"'{use_rep}' not found in adata.obsm. Run STAGATE first.")
    
    print(f"\n{'='*60}")
    print(f"Spatial Domain クラスタリング")
    print(f"{'='*60}")
    print(f"手法: {method}")
    print(f"使用する embeddings: {use_rep}")
    print(f"Embeddings 次元: {adata.obsm[use_rep].shape}")
    
    if method in ['louvain', 'leiden']:
        # 近傍グラフの構築
        print(f"近傍グラフ構築（n_neighbors={n_neighbors}）...")
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            random_state=random_state
        )
        
        # Louvain または Leiden クラスタリング
        if method == 'louvain':
            print(f"Louvain クラスタリング（resolution={resolution}）...")
            sc.tl.louvain(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=random_state
            )
        else:  # leiden
            print(f"Leiden クラスタリング（resolution={resolution}）...")
            sc.tl.leiden(
                adata,
                resolution=resolution,
                key_added=key_added,
                random_state=random_state
            )
        
        # クラスタ数
        n_detected = len(adata.obs[key_added].unique())
        print(f"検出されたクラスタ数: {n_detected}")
        
    elif method == 'mclust':
        # mclust（R の実装を使用）
        print("mclust クラスタリング...")
        try:
            from rpy2.robjects import r, pandas2ri
            from rpy2.robjects.packages import importr
            pandas2ri.activate()
            
            # R パッケージのインポート
            mclust = importr('mclust')
            
            # embeddings を取得
            embeddings = adata.obsm[use_rep]
            
            # mclust 実行
            if n_clusters is not None:
                print(f"クラスタ数を {n_clusters} に指定")
                res = mclust.Mclust(embeddings, G=n_clusters)
            else:
                print("クラスタ数を自動決定")
                res = mclust.Mclust(embeddings)
            
            # クラスタラベルを取得
            labels = np.array(res.rx2('classification')).astype(int) - 1  # 0-indexed
            adata.obs[key_added] = labels.astype(str)
            
            n_detected = len(np.unique(labels))
            print(f"検出されたクラスタ数: {n_detected}")
            
            # BIC も保存
            bic = np.array(res.rx2('BIC'))
            adata.uns[f'{key_added}_mclust_bic'] = bic
            print(f"BIC: {bic}")
            
        except ImportError:
            warnings.warn(
                "rpy2 または R の mclust パッケージが利用できません。\n"
                "Louvain を代わりに使用します。",
                UserWarning
            )
            # Fallback to Louvain
            return cluster_domains(
                adata, use_rep, method='louvain', resolution=resolution,
                key_added=key_added, n_neighbors=n_neighbors, random_state=random_state
            )
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'louvain', 'leiden', or 'mclust'.")
    
    # クラスタサイズの分布を表示
    cluster_counts = adata.obs[key_added].value_counts().sort_index()
    print("\nクラスタサイズ分布:")
    for cluster_id, count in cluster_counts.items():
        percentage = count / adata.n_obs * 100
        print(f"  Domain {cluster_id}: {count} spots ({percentage:.1f}%)")
    
    print(f"{'='*60}\n")
    
    return adata


def refine_domains_by_spatial_coherence(
    adata: AnnData,
    domain_key: str = 'domain',
    spatial_key: str = 'spatial',
    min_size: int = 10,
    key_added: str = 'domain_refined'
) -> AnnData:
    """
    空間的一貫性に基づいて domain を精緻化
    
    小さすぎる domain や孤立した domain を統合
    
    Parameters
    ----------
    adata : AnnData
        クラスタリング結果を含む AnnData
    domain_key : str, default='domain'
        元の domain ラベルのキー
    spatial_key : str, default='spatial'
        空間座標のキー
    min_size : int, default=10
        domain の最小サイズ（これより小さい場合は隣接 domain に統合）
    key_added : str, default='domain_refined'
        精緻化された domain を保存するキー
    
    Returns
    -------
    adata : AnnData
        精緻化された domain が追加された AnnData
    """
    print(f"\nDomain の空間的精緻化...")
    
    if domain_key not in adata.obs:
        raise ValueError(f"'{domain_key}' not found in adata.obs")
    
    if spatial_key not in adata.obsm:
        raise ValueError(f"'{spatial_key}' not found in adata.obsm")
    
    # TODO: 実装
    # 1. 各 domain のサイズをチェック
    # 2. 小さい domain を特定
    # 3. 空間的に最も近い domain に統合
    
    # 現時点では単純にコピー
    adata.obs[key_added] = adata.obs[domain_key].copy()
    
    print("（精緻化機能は今後実装予定）")
    
    return adata


def compute_silhouette_score(
    adata: AnnData,
    use_rep: str = 'STAGATE',
    cluster_key: str = 'domain'
) -> float:
    """
    Silhouette score を計算
    クラスタリング品質の評価指標
    
    Parameters
    ----------
    adata : AnnData
        データ
    use_rep : str, default='STAGATE'
        使用する embeddings
    cluster_key : str, default='domain'
        クラスタラベル
    
    Returns
    -------
    score : float
        Silhouette score ([-1, 1], 高いほど良い)
    """
    from sklearn.metrics import silhouette_score
    
    if use_rep not in adata.obsm:
        raise ValueError(f"'{use_rep}' not found in adata.obsm")
    
    if cluster_key not in adata.obs:
        raise ValueError(f"'{cluster_key}' not found in adata.obs")
    
    embeddings = adata.obsm[use_rep]
    labels = adata.obs[cluster_key].values
    
    score = silhouette_score(embeddings, labels)
    
    print(f"Silhouette score: {score:.4f}")
    
    return score


def compute_ari_score(
    adata: AnnData,
    pred_key: str = 'domain',
    true_key: str = 'ground_truth'
) -> float:
    """
    Adjusted Rand Index (ARI) を計算
    ground truth がある場合の評価指標
    
    Parameters
    ----------
    adata : AnnData
        データ
    pred_key : str, default='domain'
        予測ラベル
    true_key : str, default='ground_truth'
        真のラベル
    
    Returns
    -------
    ari : float
        ARI score ([0, 1], 高いほど良い)
    """
    from sklearn.metrics import adjusted_rand_score
    
    if pred_key not in adata.obs:
        raise ValueError(f"'{pred_key}' not found in adata.obs")
    
    if true_key not in adata.obs:
        raise ValueError(f"'{true_key}' not found in adata.obs")
    
    pred_labels = adata.obs[pred_key].values
    true_labels = adata.obs[true_key].values
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    
    return ari
