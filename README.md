# STAGATE 完全再現実装

**論文**: "Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder"  
**著者**: Kangning Dong & Shihua Zhang  
**出典**: Nature Communications (2022)

このリポジトリは、STAGATE (Spatial Transcriptomics Analysis with Graph Attention auto-encoder) の論文に基づく完全再現実装です。

## 📁 ディレクトリ構成

```
STAGATE_Repro/
├── README.md                      # このファイル
├── pyproject.toml                 # uv パッケージ管理
├── .venv/                         # 仮想環境（自動生成）
│
├── stagate/                       # メインパッケージ
│   ├── __init__.py
│   ├── preprocessing.py           # データ前処理（log1p, normalization, HVG選択）
│   ├── snn.py                     # Spatial Neighbor Network 構築
│   ├── celltype_snn.py            # cell type-aware SNN（Louvain pre-clustering）
│   ├── model.py                   # Graph Attention Autoencoder
│   ├── train.py                   # 学習ループ
│   ├── clustering.py              # Louvain/mclust クラスタリング
│   ├── visualization.py           # 可視化（UMAP, spatial plot, attention）
│   └── utils.py                   # ユーティリティ関数
│
├── examples/                      # 実行サンプル
│   ├── example_run.py             # 基本的な実行例
│   └── reproduce_figures.py       # 論文 Figure 2-7 の再現
│
├── data/                          # データ格納ディレクトリ（gitignore）
│   └── .gitkeep
│
└── results/                       # 結果出力ディレクトリ
    └── .gitkeep
```

## 🔬 アルゴリズム概要

### 1. データ前処理
- log1p 変換
- 正規化（各細胞の合計を10,000にスケール）
- HVG（Highly Variable Genes）3000個を選択

### 2. Spatial Neighbor Network (SNN) 構築
- **方法1**: 半径 `r` 以内の細胞を接続
- **方法2**: k-nearest neighbors（k-NN）
- 座標情報（x, y）から空間的な隣接グラフを構築

### 3. Cell type-aware SNN（オプション）
- Louvain アルゴリズムで pre-clustering（resolution=0.2）
- 異なるクラスタ間のエッジを除去（SNN pruning）
- より生物学的に妥当な spatial domain 検出

### 4. Graph Attention Autoencoder
- **Encoder**:
  - 入力次元: HVG数（通常3000）
  - 隠れ層: 512次元
  - 出力: 30次元の潜在表現
  - 活性化関数: ELU
  
- **Attention Layer**（論文 Eq.5-7）:
  ```
  e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
  α_ij = softmax_j(e_ij)
  h'_i = σ(Σ_j α_ij W h_j)
  ```

- **Decoder**:
  - 重み共有: W^b(k) = W(k)^T
  - 元の遺伝子発現を再構成

### 5. 学習
- **Optimizer**: Adam
- **Learning rate**: 1e-4
- **Weight decay**: 1e-4
- **Iterations**: 500（小規模データ）/ 1000（大規模データ）
- **Loss**: Reconstruction loss (L2)

### 6. クラスタリング
- 学習済みの30次元潜在表現を使用
- Louvain アルゴリズムまたは mclust
- spatial domain の同定

## 🚀 使用方法

### インストール

```bash
# uvがインストールされていない場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 基本的な実行例

```python
from stagate import STAGATE
import scanpy as sc

# データ読み込み
adata = sc.read_h5ad("data/your_spatial_data.h5ad")

# STAGATE実行
model = STAGATE(
    adata=adata,
    spatial_key="spatial",
    n_epochs=1000,
    lr=1e-4,
    weight_decay=1e-4,
    hidden_dim=512,
    latent_dim=30,
    use_celltype_snn=True,
    louvain_resolution=0.2
)

# 学習
model.train()

# クラスタリング
model.clustering(method='louvain', resolution=0.5)

# 可視化
model.plot_spatial_domains()
model.plot_umap()
model.plot_attention_weights()
```

### コマンドラインからの実行

```bash
# 基本的な実行
uv run python examples/example_run.py --input data/sample.h5ad --output results/

# 論文の図を再現
uv run python examples/reproduce_figures.py --dataset DLPFC
```

## 📊 再現実験

論文の主要な Figure を再現するスクリプトを提供しています：

- **Figure 2**: STAGATE の概要と性能比較
- **Figure 3**: DLPFC データでの spatial domain 検出
- **Figure 4**: Attention weights の可視化
- **Figure 5**: Cell type-aware SNN の効果
- **Figure 6**: 複数データセットでの評価
- **Figure 7**: Ablation study

```bash
uv run python examples/reproduce_figures.py --figure 3
```

## 🔧 ハイパーパラメータ（論文準拠）

| パラメータ | デフォルト値 | 説明 |
|----------|------------|------|
| `hidden_dim` | 512 | Encoder の隠れ層次元 |
| `latent_dim` | 30 | 潜在表現の次元 |
| `lr` | 1e-4 | 学習率 |
| `weight_decay` | 1e-4 | Weight decay |
| `n_epochs` | 1000 | 学習エポック数 |
| `n_hvgs` | 3000 | 使用する HVG 数 |
| `radius` | 150 | SNN の半径（μm） |
| `k_neighbors` | None | k-NN の k（radius と排他） |
| `louvain_resolution` | 0.2 | Pre-clustering の resolution |

## 📚 必要なパッケージ

- Python >= 3.8
- torch >= 2.0
- torch-geometric >= 2.3
- scanpy >= 1.9
- anndata >= 0.8
- numpy
- scipy
- pandas
- matplotlib
- seaborn
- scikit-learn
- umap-learn

すべて `uv sync` で自動インストールされます。

## 📖 引用

```bibtex
@article{dong2022stagate,
  title={Deciphering spatial domains from spatially resolved transcriptomics with an adaptive graph attention auto-encoder},
  author={Dong, Kangning and Zhang, Shihua},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```

## 📝 ライセンス

MIT License

## 🤝 貢献

Issue や Pull Request を歓迎します。

## 📧 連絡先

実装に関する質問は Issue でお願いします。
