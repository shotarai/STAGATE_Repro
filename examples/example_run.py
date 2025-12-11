"""
STAGATE 基本実行例
"""

import argparse
import scanpy as sc
from pathlib import Path
import torch

# STAGATE モジュールのインポート
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from stagate import (
    preprocess_data,
    build_spatial_network,
    build_celltype_aware_snn,
    train_stagate,
    cluster_domains,
    plot_spatial_domains,
    plot_umap,
    set_seed
)


def main(args):
    """
    STAGATE の基本的な実行フロー
    
    フロー:
    1. データ読み込み
    2. 前処理（log1p, normalization, HVG selection）
    3. SNN 構築
    4. （オプション）cell type-aware SNN
    5. STAGATE 学習
    6. クラスタリング
    7. 可視化
    8. 結果保存
    """
    # 乱数シード設定
    set_seed(args.seed)
    
    # デバイス設定
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    
    print(f"\n{'='*70}")
    print("STAGATE 実行開始")
    print(f"{'='*70}")
    print(f"入力ファイル: {args.input}")
    print(f"出力ディレクトリ: {args.output}")
    print(f"デバイス: {device}")
    print(f"{'='*70}\n")
    
    # ========== 1. データ読み込み ==========
    print("ステップ 1/7: データ読み込み")
    adata = sc.read_h5ad(args.input)
    print(f"データサイズ: {adata.shape}")
    print(f"スポット数: {adata.n_obs}, 遺伝子数: {adata.n_vars}")
    
    # ========== 2. 前処理 ==========
    print("\nステップ 2/7: データ前処理")
    preprocess_data(
        adata,
        n_top_genes=args.n_hvgs,
        target_sum=1e4,
        log_transform=True,
        copy=False
    )
    
    # ========== 3. SNN 構築 ==========
    print("\nステップ 3/7: Spatial Neighbor Network (SNN) 構築")
    edge_index, edge_attr = build_spatial_network(
        adata,
        spatial_key=args.spatial_key,
        radius=args.radius,
        k_neighbors=args.k_neighbors,
        method=args.snn_method,
        return_adjacency=True
    )
    
    # ========== 4. Cell type-aware SNN（オプション）==========
    if args.use_celltype_snn:
        print("\nステップ 4/7: Cell type-aware SNN 構築")
        edge_index, edge_attr = build_celltype_aware_snn(
            adata,
            edge_index,
            edge_attr,
            louvain_resolution=args.louvain_resolution,
            use_rep='X',
            n_neighbors=15,
            copy=False
        )
    else:
        print("\nステップ 4/7: Cell type-aware SNN をスキップ")
    
    # ========== 5. STAGATE 学習 ==========
    print("\nステップ 5/7: STAGATE 学習")
    model, embeddings, history = train_stagate(
        adata,
        edge_index,
        edge_attr,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_epochs=args.n_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        heads=args.heads,
        dropout=args.dropout,
        device=device,
        verbose=True,
        random_state=args.seed
    )
    
    # ========== 6. クラスタリング ==========
    print("\nステップ 6/7: Spatial domain クラスタリング")
    cluster_domains(
        adata,
        use_rep='STAGATE',
        method=args.cluster_method,
        resolution=args.cluster_resolution,
        key_added='domain',
        n_neighbors=15,
        random_state=args.seed
    )
    
    # ========== 7. 可視化 ==========
    print("\nステップ 7/7: 可視化")
    
    # 出力ディレクトリの作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Spatial domain プロット
    print("  - Spatial domain プロット")
    plot_spatial_domains(
        adata,
        domain_key='domain',
        spatial_key=args.spatial_key,
        spot_size=args.spot_size,
        save=output_dir / 'spatial_domains.png'
    )
    
    # UMAP プロット
    print("  - UMAP プロット")
    plot_umap(
        adata,
        color=['domain'],
        use_rep='STAGATE',
        save=output_dir / 'umap.png'
    )
    
    # ========== 結果保存 ==========
    print("\n結果を保存中...")
    
    # AnnData 保存
    output_h5ad = output_dir / 'stagate_result.h5ad'
    adata.write_h5ad(output_h5ad)
    print(f"  - AnnData を保存: {output_h5ad}")
    
    # モデル保存
    if args.save_model:
        from stagate.utils import save_model
        model_path = output_dir / 'stagate_model.pt'
        save_model(model, str(model_path))
        print(f"  - モデルを保存: {model_path}")
    
    # 学習履歴プロット
    from stagate.train import plot_training_history
    plot_training_history(history, save=output_dir / 'training_history.png')
    print(f"  - 学習履歴を保存: {output_dir / 'training_history.png'}")
    
    print(f"\n{'='*70}")
    print("STAGATE 実行完了!")
    print(f"結果は {output_dir} に保存されました")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STAGATE 実行スクリプト')
    
    # 入出力
    parser.add_argument('--input', type=str, required=True,
                        help='入力データ (.h5ad)')
    parser.add_argument('--output', type=str, default='./results',
                        help='出力ディレクトリ')
    parser.add_argument('--spatial-key', type=str, default='spatial',
                        help='空間座標のキー（adata.obsm）')
    
    # 前処理
    parser.add_argument('--n-hvgs', type=int, default=3000,
                        help='HVG の数')
    
    # SNN
    parser.add_argument('--snn-method', type=str, default='radius',
                        choices=['radius', 'knn'],
                        help='SNN 構築方法')
    parser.add_argument('--radius', type=float, default=None,
                        help='SNN の半径（method=radius の場合）')
    parser.add_argument('--k-neighbors', type=int, default=None,
                        help='k-NN の k（method=knn の場合）')
    parser.add_argument('--use-celltype-snn', action='store_true',
                        help='Cell type-aware SNN を使用')
    parser.add_argument('--louvain-resolution', type=float, default=0.2,
                        help='Pre-clustering の resolution')
    
    # モデル
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='隠れ層の次元')
    parser.add_argument('--latent-dim', type=int, default=30,
                        help='潜在層の次元')
    parser.add_argument('--heads', type=int, default=1,
                        help='Multi-head attention のヘッド数')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout 率')
    
    # 学習
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='学習エポック数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学習率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--use-gpu', action='store_true',
                        help='GPU を使用')
    
    # クラスタリング
    parser.add_argument('--cluster-method', type=str, default='louvain',
                        choices=['louvain', 'leiden', 'mclust'],
                        help='クラスタリング手法')
    parser.add_argument('--cluster-resolution', type=float, default=0.5,
                        help='クラスタリングの resolution')
    
    # 可視化
    parser.add_argument('--spot-size', type=float, default=1.0,
                        help='プロットのスポットサイズ')
    
    # その他
    parser.add_argument('--seed', type=int, default=0,
                        help='乱数シード')
    parser.add_argument('--save-model', action='store_true',
                        help='モデルを保存')
    
    args = parser.parse_args()
    main(args)
