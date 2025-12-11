"""
テスト用のサンプルデータ生成スクリプト

STAGATE の動作確認用に、合成データを生成します。
"""

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc
from pathlib import Path


def generate_synthetic_spatial_data(
    n_spots: int = 500,
    n_genes: int = 2000,
    n_domains: int = 5,
    grid_size: int = 30,
    noise_level: float = 0.3,
    seed: int = 0
) -> AnnData:
    """
    合成空間トランスクリプトームデータを生成
    
    Parameters
    ----------
    n_spots : int
        スポット数
    n_genes : int
        遺伝子数
    n_domains : int
        Spatial domain 数
    grid_size : int
        グリッドサイズ（空間座標の範囲）
    noise_level : float
        ノイズレベル
    seed : int
        乱数シード
    
    Returns
    -------
    adata : AnnData
        合成データ
    """
    np.random.seed(seed)
    
    print(f"合成データ生成中...")
    print(f"  - スポット数: {n_spots}")
    print(f"  - 遺伝子数: {n_genes}")
    print(f"  - Domain 数: {n_domains}")
    
    # ========== 空間座標の生成 ==========
    # グリッド上にランダムに配置
    coords = np.random.uniform(0, grid_size, size=(n_spots, 2))
    
    # ========== Domain ラベルの生成 ==========
    # 空間的にクラスタ化された domain を生成
    domain_centers = np.random.uniform(0, grid_size, size=(n_domains, 2))
    
    # 各スポットを最も近い domain center に割り当て
    from scipy.spatial.distance import cdist
    distances = cdist(coords, domain_centers)
    domain_labels = np.argmin(distances, axis=1)
    
    # ========== 遺伝子発現の生成 ==========
    # 各 domain に特異的な遺伝子発現パターンを生成
    
    # Domain-specific genes
    n_genes_per_domain = n_genes // n_domains
    expression = np.zeros((n_spots, n_genes))
    
    for domain_id in range(n_domains):
        # このdomain に属するスポット
        spot_mask = domain_labels == domain_id
        n_spots_in_domain = spot_mask.sum()
        
        # Domain-specific genes の範囲
        gene_start = domain_id * n_genes_per_domain
        gene_end = (domain_id + 1) * n_genes_per_domain
        
        # 高発現パターン（負の二項分布）
        for i in range(gene_start, min(gene_end, n_genes)):
            # Domain 内で高発現
            expression[spot_mask, i] = np.random.negative_binomial(
                n=10, p=0.3, size=n_spots_in_domain
            )
            # Domain 外で低発現
            expression[~spot_mask, i] = np.random.negative_binomial(
                n=2, p=0.8, size=n_spots - n_spots_in_domain
            )
    
    # 共通遺伝子（background）
    remaining_genes = n_genes - n_domains * n_genes_per_domain
    if remaining_genes > 0:
        expression[:, -remaining_genes:] = np.random.negative_binomial(
            n=5, p=0.5, size=(n_spots, remaining_genes)
        )
    
    # ノイズ追加
    noise = np.random.normal(0, noise_level * expression.mean(), size=expression.shape)
    expression = np.maximum(0, expression + noise)
    
    # ========== AnnData オブジェクトの作成 ==========
    var_names = [f"Gene_{i}" for i in range(n_genes)]
    obs_names = [f"Spot_{i}" for i in range(n_spots)]
    
    adata = AnnData(
        X=expression,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names)
    )
    
    # 空間座標を追加
    adata.obsm['spatial'] = coords
    
    # Ground truth domain を追加
    adata.obs['ground_truth'] = pd.Categorical([f"Domain_{d}" for d in domain_labels])
    
    # メタデータ
    adata.uns['spatial'] = {
        'grid_size': grid_size,
        'n_domains': n_domains,
        'domain_centers': domain_centers
    }
    
    print(f"合成データ生成完了!")
    print(f"  - データサイズ: {adata.shape}")
    
    return adata


def main():
    """メイン関数"""
    
    # 出力ディレクトリ
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # サンプルデータ生成
    print("\n" + "="*70)
    print("サンプルデータ生成")
    print("="*70 + "\n")
    
    # 小規模データ（テスト用）
    print("1. 小規模データ（テスト用）")
    adata_small = generate_synthetic_spatial_data(
        n_spots=200,
        n_genes=500,
        n_domains=3,
        grid_size=20,
        seed=0
    )
    
    output_path = output_dir / 'sample_small.h5ad'
    adata_small.write_h5ad(output_path)
    print(f"保存しました: {output_path}\n")
    
    # 中規模データ（標準）
    print("2. 中規模データ（標準）")
    adata_medium = generate_synthetic_spatial_data(
        n_spots=1000,
        n_genes=2000,
        n_domains=5,
        grid_size=40,
        seed=1
    )
    
    output_path = output_dir / 'sample_medium.h5ad'
    adata_medium.write_h5ad(output_path)
    print(f"保存しました: {output_path}\n")
    
    # 大規模データ（ベンチマーク用）
    print("3. 大規模データ（ベンチマーク用）")
    adata_large = generate_synthetic_spatial_data(
        n_spots=3000,
        n_genes=3000,
        n_domains=7,
        grid_size=60,
        seed=2
    )
    
    output_path = output_dir / 'sample_large.h5ad'
    adata_large.write_h5ad(output_path)
    print(f"保存しました: {output_path}\n")
    
    print("="*70)
    print("サンプルデータ生成完了!")
    print(f"データは {output_dir} に保存されました")
    print("="*70 + "\n")
    
    # 使用例を表示
    print("使用例:")
    print(f"  uv run python examples/example_run.py --input {output_dir}/sample_small.h5ad --output results/test")


if __name__ == '__main__':
    main()
