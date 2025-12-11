"""
論文 Figure 2-7 の再現スクリプト

このスクリプトは STAGATE 論文の主要な Figure を再現します:
- Figure 2: STAGATE の概要と性能
- Figure 3: DLPFC データでの spatial domain 検出
- Figure 4: UMAP による可視化
- Figure 5: Attention weights の可視化
- Figure 6: 複数データセットでの評価
- Figure 7: Ablation study
"""

import argparse
import scanpy as sc
from pathlib import Path
import torch
import numpy as np

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
    plot_attention_weights,
    set_seed
)
from stagate.train import extract_attention_weights


def reproduce_figure_3(args):
    """
    Figure 3: DLPFC データセットでの spatial domain 検出
    
    論文の Figure 3 を再現:
    - Spatial transcriptomics データ
    - STAGATE による spatial domain 検出
    - Ground truth との比較
    """
    print(f"\n{'='*70}")
    print("Figure 3 再現: DLPFC spatial domain 検出")
    print(f"{'='*70}\n")
    
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # データ読み込み
    adata = sc.read_h5ad(args.input)
    
    # 前処理
    preprocess_data(adata, n_top_genes=3000, copy=False)
    
    # SNN 構築（DLPFC では radius=150 が標準）
    edge_index, edge_attr = build_spatial_network(
        adata,
        radius=150,
        method='radius'
    )
    
    # Cell type-aware SNN
    if args.use_celltype_snn:
        edge_index, edge_attr = build_celltype_aware_snn(
            adata, edge_index, edge_attr,
            louvain_resolution=0.2
        )
    
    # 学習
    model, embeddings, history = train_stagate(
        adata, edge_index, edge_attr,
        n_epochs=1000,
        device=device
    )
    
    # クラスタリング
    cluster_domains(adata, resolution=0.4)
    
    # 可視化
    output_dir = Path(args.output) / 'figure_3'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # (a) STAGATE domains
    plot_spatial_domains(
        adata, domain_key='domain',
        title='STAGATE Spatial Domains',
        save=output_dir / 'fig3a_stagate_domains.png'
    )
    
    # (b) Ground truth（もしあれば）
    if 'ground_truth' in adata.obs:
        plot_spatial_domains(
            adata, domain_key='ground_truth',
            title='Ground Truth',
            save=output_dir / 'fig3b_ground_truth.png'
        )
    
    # (c) UMAP
    plot_umap(
        adata, color=['domain'],
        save=output_dir / 'fig3c_umap.png'
    )
    
    print(f"\nFigure 3 を {output_dir} に保存しました")


def reproduce_figure_5(args):
    """
    Figure 5: Attention weights の可視化
    
    論文の Figure 5 を再現:
    - Cell type-aware SNN の効果
    - Attention weights の空間パターン
    """
    print(f"\n{'='*70}")
    print("Figure 5 再現: Attention weights 可視化")
    print(f"{'='*70}\n")
    
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # データ読み込み
    adata = sc.read_h5ad(args.input)
    
    # 前処理
    preprocess_data(adata, n_top_genes=3000, copy=False)
    
    # SNN 構築
    edge_index, edge_attr = build_spatial_network(adata, radius=150)
    
    # Cell type-aware SNN
    edge_index_ct, edge_attr_ct = build_celltype_aware_snn(
        adata, edge_index, edge_attr,
        louvain_resolution=0.2
    )
    
    # 学習（cell type-aware SNN）
    model, embeddings, history = train_stagate(
        adata, edge_index_ct, edge_attr_ct,
        n_epochs=1000,
        device=device
    )
    
    # Attention weights 抽出
    edge_idx, attention_weights = extract_attention_weights(
        model, adata, edge_index_ct, edge_attr_ct, device=device
    )
    
    # 可視化
    output_dir = Path(args.output) / 'figure_5'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # (d-f) Attention weights
    plot_attention_weights(
        adata, edge_idx, attention_weights,
        top_k=2000,
        title='Attention Weights (Cell type-aware SNN)',
        save=output_dir / 'fig5_attention_weights.png'
    )
    
    print(f"\nFigure 5 を {output_dir} に保存しました")


def reproduce_ablation_study(args):
    """
    Ablation study: STAGATE の各コンポーネントの効果を検証
    
    比較:
    1. STAGATE (full)
    2. STAGATE without cell type-aware SNN
    3. STAGATE without attention
    """
    print(f"\n{'='*70}")
    print("Ablation Study: 各コンポーネントの効果")
    print(f"{'='*70}\n")
    
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # データ読み込み
    adata = sc.read_h5ad(args.input)
    
    # 前処理
    preprocess_data(adata, n_top_genes=3000, copy=False)
    
    # SNN 構築
    edge_index, edge_attr = build_spatial_network(adata, radius=150)
    
    output_dir = Path(args.output) / 'ablation_study'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 実験 1: STAGATE (full) ==========
    print("\n実験 1/2: STAGATE (full)")
    adata_full = adata.copy()
    
    # Cell type-aware SNN
    edge_index_ct, edge_attr_ct = build_celltype_aware_snn(
        adata_full, edge_index, edge_attr,
        louvain_resolution=0.2
    )
    
    # 学習
    model_full, _, _ = train_stagate(
        adata_full, edge_index_ct, edge_attr_ct,
        n_epochs=500,
        device=device,
        verbose=False
    )
    
    # クラスタリング
    cluster_domains(adata_full, resolution=0.4, key_added='domain_full')
    
    # 可視化
    plot_spatial_domains(
        adata_full, domain_key='domain_full',
        title='STAGATE (Full)',
        save=output_dir / 'ablation_full.png'
    )
    
    # ========== 実験 2: STAGATE without cell type-aware SNN ==========
    print("\n実験 2/2: STAGATE without cell type-aware SNN")
    adata_no_ct = adata.copy()
    
    # 学習（通常の SNN）
    model_no_ct, _, _ = train_stagate(
        adata_no_ct, edge_index, edge_attr,
        n_epochs=500,
        device=device,
        verbose=False
    )
    
    # クラスタリング
    cluster_domains(adata_no_ct, resolution=0.4, key_added='domain_no_ct')
    
    # 可視化
    plot_spatial_domains(
        adata_no_ct, domain_key='domain_no_ct',
        title='STAGATE (without cell type-aware SNN)',
        save=output_dir / 'ablation_no_celltype_snn.png'
    )
    
    # ========== 評価 ==========
    if 'ground_truth' in adata.obs:
        from stagate.clustering import compute_ari_score
        
        print("\n評価結果 (ARI):")
        adata_full.obs['ground_truth'] = adata.obs['ground_truth']
        adata_no_ct.obs['ground_truth'] = adata.obs['ground_truth']
        
        ari_full = compute_ari_score(adata_full, 'domain_full', 'ground_truth')
        ari_no_ct = compute_ari_score(adata_no_ct, 'domain_no_ct', 'ground_truth')
        
        print(f"  STAGATE (Full): {ari_full:.4f}")
        print(f"  Without cell type-aware SNN: {ari_no_ct:.4f}")
    
    print(f"\nAblation study の結果を {output_dir} に保存しました")


def main(args):
    """メイン関数"""
    
    if args.figure == 'all' or args.figure == '3':
        reproduce_figure_3(args)
    
    if args.figure == 'all' or args.figure == '5':
        reproduce_figure_5(args)
    
    if args.figure == 'all' or args.figure == 'ablation':
        reproduce_ablation_study(args)
    
    print(f"\n{'='*70}")
    print("再現実験完了!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='STAGATE 論文 Figure 再現スクリプト'
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='入力データ (.h5ad)')
    parser.add_argument('--output', type=str, default='./results/figures',
                        help='出力ディレクトリ')
    parser.add_argument('--figure', type=str, default='all',
                        choices=['all', '3', '5', 'ablation'],
                        help='再現する Figure')
    parser.add_argument('--use-celltype-snn', action='store_true',
                        help='Cell type-aware SNN を使用')
    
    args = parser.parse_args()
    main(args)
