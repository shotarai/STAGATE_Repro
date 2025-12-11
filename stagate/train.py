"""
学習モジュール
論文 Methods の学習手順を厳密に実装
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from anndata import AnnData
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import time

from .model import GraphAttentionAutoencoder


def train_stagate(
    adata: AnnData,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    hidden_dim: int = 512,
    latent_dim: int = 30,
    n_epochs: int = 1000,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    heads: int = 1,
    dropout: float = 0.0,
    device: str = 'cpu',
    verbose: bool = True,
    random_state: int = 0
) -> Tuple[GraphAttentionAutoencoder, np.ndarray, Dict]:
    """
    STAGATE の学習を実行
    
    論文 Methods の記載:
    - Optimizer: Adam
    - Learning rate: 1e-4
    - Weight decay: 1e-4
    - Epochs: 500 (small datasets) / 1000 (large datasets)
    - Loss: Reconstruction loss (MSE / L2)
    
    Parameters
    ----------
    adata : AnnData
        前処理済みの AnnData オブジェクト
    edge_index : torch.Tensor, shape (2, num_edges)
        SNN のエッジインデックス
    edge_attr : torch.Tensor, optional
        エッジの重み
    hidden_dim : int, default=512
        隠れ層の次元（論文では 512）
    latent_dim : int, default=30
        潜在表現の次元（論文では 30）
    n_epochs : int, default=1000
        学習エポック数
    lr : float, default=1e-4
        学習率（論文では 1e-4）
    weight_decay : float, default=1e-4
        Weight decay（論文では 1e-4）
    heads : int, default=1
        Multi-head attention のヘッド数
    dropout : float, default=0.0
        Dropout 率
    device : str, default='cpu'
        使用デバイス（'cpu' or 'cuda'）
    verbose : bool, default=True
        進捗を表示するかどうか
    random_state : int, default=0
        乱数シード
    
    Returns
    -------
    model : GraphAttentionAutoencoder
        学習済みモデル
    embeddings : np.ndarray, shape (n_spots, latent_dim)
        潜在表現（STAGATE embeddings）
    history : dict
        学習履歴（loss など）
    """
    # 乱数シードの設定
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # デバイスの設定
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA が利用できないため、CPU を使用します")
        device = 'cpu'
    device = torch.device(device)
    
    print(f"\n{'='*60}")
    print("STAGATE 学習開始")
    print(f"{'='*60}")
    print(f"デバイス: {device}")
    print(f"スポット数: {adata.n_obs}")
    print(f"遺伝子数（HVG）: {adata.n_vars}")
    print(f"エッジ数: {edge_index.shape[1]}")
    print(f"エポック数: {n_epochs}")
    print(f"学習率: {lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"隠れ層次元: {hidden_dim}")
    print(f"潜在層次元: {latent_dim}")
    print(f"{'='*60}\n")
    
    # データの準備
    # AnnData.X を torch.Tensor に変換
    if hasattr(adata.X, 'toarray'):
        # Sparse matrix の場合
        x = torch.FloatTensor(adata.X.toarray()).to(device)
    else:
        # Dense matrix の場合
        x = torch.FloatTensor(adata.X).to(device)
    
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    
    # モデルの初期化
    input_dim = adata.n_vars
    model = GraphAttentionAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    # Optimizer の設定（論文: Adam, lr=1e-4, weight_decay=1e-4）
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Loss function: MSE (L2 reconstruction loss)
    criterion = nn.MSELoss()
    
    # 学習履歴
    history = {
        'loss': [],
        'epoch_time': []
    }
    
    # 学習ループ
    model.train()
    
    if verbose:
        pbar = tqdm(range(n_epochs), desc="Training")
    else:
        pbar = range(n_epochs)
    
    for epoch in pbar:
        start_time = time.time()
        
        # Forward pass
        x_recon, z = model(x, edge_index, edge_attr)
        
        # Loss 計算（Reconstruction loss）
        loss = criterion(x_recon, x)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 学習履歴の記録
        epoch_time = time.time() - start_time
        history['loss'].append(loss.item())
        history['epoch_time'].append(epoch_time)
        
        # 進捗表示
        if verbose:
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'time': f'{epoch_time:.2f}s'
            })
        
        # 定期的にログ出力
        if verbose and (epoch + 1) % 100 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Time: {epoch_time:.2f}s")
    
    # 学習完了
    print(f"\n{'='*60}")
    print("学習完了!")
    print(f"最終 Loss: {history['loss'][-1]:.6f}")
    print(f"平均エポック時間: {np.mean(history['epoch_time']):.2f}s")
    print(f"総学習時間: {np.sum(history['epoch_time']):.2f}s")
    print(f"{'='*60}\n")
    
    # 潜在表現（embeddings）を抽出
    model.eval()
    with torch.no_grad():
        _, z = model(x, edge_index, edge_attr)
        embeddings = z.cpu().numpy()
    
    # AnnData に保存
    adata.obsm['STAGATE'] = embeddings
    
    return model, embeddings, history


def extract_embeddings(
    model: GraphAttentionAutoencoder,
    adata: AnnData,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> np.ndarray:
    """
    学習済みモデルから embeddings を抽出
    
    Parameters
    ----------
    model : GraphAttentionAutoencoder
        学習済みモデル
    adata : AnnData
        データ
    edge_index : torch.Tensor
        エッジインデックス
    edge_attr : torch.Tensor, optional
        エッジの重み
    device : str
        デバイス
    
    Returns
    -------
    embeddings : np.ndarray, shape (n_spots, latent_dim)
        潜在表現
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    # データの準備
    if hasattr(adata.X, 'toarray'):
        x = torch.FloatTensor(adata.X.toarray()).to(device)
    else:
        x = torch.FloatTensor(adata.X).to(device)
    
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    
    # Forward pass
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_attr)
        embeddings = z.cpu().numpy()
    
    return embeddings


def extract_attention_weights(
    model: GraphAttentionAutoencoder,
    adata: AnnData,
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attention weights を抽出（可視化用）
    
    Parameters
    ----------
    model : GraphAttentionAutoencoder
        学習済みモデル
    adata : AnnData
        データ
    edge_index : torch.Tensor
        エッジインデックス
    edge_attr : torch.Tensor, optional
        エッジの重み
    device : str
        デバイス
    
    Returns
    -------
    edge_index_np : np.ndarray, shape (2, num_edges)
        エッジインデックス
    attention_weights : np.ndarray, shape (num_edges,)
        各エッジの attention weight
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    
    # データの準備
    if hasattr(adata.X, 'toarray'):
        x = torch.FloatTensor(adata.X.toarray()).to(device)
    else:
        x = torch.FloatTensor(adata.X).to(device)
    
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    
    # Forward pass with attention weights
    with torch.no_grad():
        _, attention_weights = model.encode(
            x, edge_index, edge_attr, return_attention_weights=True
        )
    
    # attention_weights: (edge_index, alpha)
    edge_index_np = attention_weights[0].cpu().numpy()
    alpha = attention_weights[1].cpu().numpy()
    
    return edge_index_np, alpha


def plot_training_history(history: Dict, save: Optional[str] = None):
    """
    学習履歴をプロット
    
    Parameters
    ----------
    history : dict
        学習履歴
    save : str, optional
        保存先のパス
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss の推移
    ax = axes[0]
    ax.plot(history['loss'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # エポック時間の推移
    ax = axes[1]
    ax.plot(history['epoch_time'], linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Epoch Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f"図を保存しました: {save}")
    
    plt.show()
