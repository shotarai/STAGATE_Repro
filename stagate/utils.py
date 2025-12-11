"""
ユーティリティ関数
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 0):
    """
    再現性のため、すべての乱数シードを設定
    
    Parameters
    ----------
    seed : int, default=0
        乱数シード
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"乱数シードを {seed} に設定しました")


def get_device(device: Optional[str] = None) -> torch.device:
    """
    使用するデバイスを取得
    
    Parameters
    ----------
    device : str, optional
        'cpu', 'cuda', 'cuda:0' など
        指定しない場合は自動判定
    
    Returns
    -------
    device : torch.device
        使用するデバイス
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    if device.type == 'cuda':
        print(f"使用デバイス: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"使用デバイス: {device}")
    
    return device


def print_memory_usage():
    """
    GPU メモリ使用量を表示
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU メモリ使用量: {allocated:.2f} GB (allocated), {reserved:.2f} GB (reserved)")


def save_model(model: torch.nn.Module, path: str):
    """
    モデルを保存
    
    Parameters
    ----------
    model : torch.nn.Module
        保存するモデル
    path : str
        保存先のパス (.pt または .pth)
    """
    torch.save(model.state_dict(), path)
    print(f"モデルを保存しました: {path}")


def load_model(model: torch.nn.Module, path: str, device: str = 'cpu'):
    """
    モデルを読み込み
    
    Parameters
    ----------
    model : torch.nn.Module
        読み込み先のモデル（アーキテクチャが一致している必要がある）
    path : str
        モデルのパス (.pt または .pth)
    device : str, default='cpu'
        デバイス
    
    Returns
    -------
    model : torch.nn.Module
        読み込まれたモデル
    """
    device = torch.device(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"モデルを読み込みました: {path}")
    return model


def compute_pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """
    座標間のペアワイズ距離を計算
    
    Parameters
    ----------
    coords : np.ndarray, shape (n, d)
        座標（n 個の d 次元ベクトル）
    
    Returns
    -------
    distances : np.ndarray, shape (n, n)
        距離行列
    """
    from scipy.spatial.distance import cdist
    return cdist(coords, coords, metric='euclidean')


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    隣接行列を正規化（対称正規化）
    
    D^{-1/2} A D^{-1/2}
    
    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        隣接行列
    
    Returns
    -------
    adj_norm : np.ndarray, shape (n, n)
        正規化された隣接行列
    """
    # 次数行列
    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    
    # 対角行列に変換
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    
    # 正規化
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return adj_norm
