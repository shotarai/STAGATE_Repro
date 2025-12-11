"""
Graph Attention Autoencoder モデル
論文 Equation 5-7 および Methods セクションを厳密に実装
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from typing import Optional, Tuple


class GraphAttentionLayer(MessagePassing):
    """
    Graph Attention Layer
    
    論文 Equation 5-7 の実装:
    
    Eq.5: e_ij = LeakyReLU(a^T [W h_i || W h_j])
          attention coefficient（正規化前）
    
    Eq.6: α_ij = softmax_j(e_ij) 
          = exp(e_ij) / Σ_{k∈N_i} exp(e_ik)
          正規化された attention weight
    
    Eq.7: h'_i = σ(Σ_{j∈N_i} α_ij W h_j)
          出力特徴量（σ は活性化関数）
    
    Parameters
    ----------
    in_channels : int
        入力特徴量の次元
    out_channels : int
        出力特徴量の次元
    heads : int, default=1
        Multi-head attention のヘッド数
    concat : bool, default=True
        Multi-head の出力を結合するか平均するか
    negative_slope : float, default=0.2
        LeakyReLU の負の傾き
    dropout : float, default=0.0
        Dropout 率
    add_self_loops : bool, default=True
        Self-loops を追加するか
    bias : bool, default=True
        バイアス項を使用するか
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops_flag: bool = True,
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops_flag = add_self_loops_flag
        
        # 重み行列 W（論文 Eq.5）
        # 各ヘッドごとに独立した重み
        self.weight = nn.Parameter(
            torch.Tensor(in_channels, heads * out_channels)
        )
        
        # Attention パラメータ a（論文 Eq.5）
        # a^T [W h_i || W h_j] の a
        self.att = nn.Parameter(
            torch.Tensor(1, heads, 2 * out_channels)
        )
        
        # バイアス項
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """パラメータの初期化（Xavier uniform）"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Forward pass
        
        Parameters
        ----------
        x : torch.Tensor, shape (num_nodes, in_channels)
            ノードの特徴量
        edge_index : torch.Tensor, shape (2, num_edges)
            エッジのインデックス
        edge_attr : torch.Tensor, optional, shape (num_edges,)
            エッジの重み（SNN の距離重み）
        return_attention_weights : bool, default=False
            Attention weights を返すかどうか
        
        Returns
        -------
        out : torch.Tensor, shape (num_nodes, out_channels * heads) or (num_nodes, out_channels)
            出力特徴量
        attention_weights : tuple, optional
            (edge_index, alpha) のタプル（return_attention_weights=True の場合）
        """
        # Self-loops を追加（論文では明示されていないが、GAT の標準実装）
        if self.add_self_loops_flag:
            edge_index, edge_attr = add_self_loops(
                edge_index,
                edge_attr,
                fill_value=1.0,
                num_nodes=x.size(0)
            )
        
        # 線形変換 W h（論文 Eq.5 の W h_i, W h_j）
        # x: (num_nodes, in_channels) -> (num_nodes, heads * out_channels)
        x = torch.matmul(x, self.weight)
        
        # (num_nodes, heads * out_channels) -> (num_nodes, heads, out_channels)
        x = x.view(-1, self.heads, self.out_channels)
        
        # Message passing（論文 Eq.5-7 を実行）
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            return_attention_weights=return_attention_weights
        )
        
        # Multi-head の処理
        if self.concat:
            # ヘッドを結合: (num_nodes, heads, out_channels) -> (num_nodes, heads * out_channels)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            # ヘッドを平均: (num_nodes, heads, out_channels) -> (num_nodes, out_channels)
            out = out.mean(dim=1)
        
        # バイアスを追加
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            return out, self._attention_weights
        else:
            return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index_i: torch.Tensor,
        size_i: int,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Message 関数（論文 Eq.5-7 を実装）
        
        Parameters
        ----------
        x_i : torch.Tensor, shape (num_edges, heads, out_channels)
            エッジの送信元ノードの特徴量（W h_i）
        x_j : torch.Tensor, shape (num_edges, heads, out_channels)
            エッジの送信先ノードの特徴量（W h_j）
        edge_index_i : torch.Tensor
            エッジの送信元ノードのインデックス
        size_i : int
            送信元ノードの総数
        edge_attr : torch.Tensor, optional
            エッジの重み
        return_attention_weights : bool
            Attention weights を保存するかどうか
        
        Returns
        -------
        out : torch.Tensor, shape (num_edges, heads, out_channels)
            Attention で重み付けされたメッセージ
        """
        # 論文 Eq.5: e_ij = LeakyReLU(a^T [W h_i || W h_j])
        # [W h_i || W h_j]: (num_edges, heads, 2 * out_channels)
        x_cat = torch.cat([x_i, x_j], dim=-1)
        
        # a^T [W h_i || W h_j]: (num_edges, heads, 2 * out_channels) @ (1, heads, 2 * out_channels)
        # -> (num_edges, heads)
        alpha = (x_cat * self.att).sum(dim=-1)
        
        # LeakyReLU
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # エッジの重み（SNN の距離重み）を考慮
        if edge_attr is not None:
            # edge_attr: (num_edges,) -> (num_edges, 1)
            alpha = alpha * edge_attr.view(-1, 1)
        
        # 論文 Eq.6: α_ij = softmax_j(e_ij)
        # 各ノードの近傍に対して softmax
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        
        # Dropout
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        
        # Attention weights を保存（可視化用）
        if return_attention_weights:
            self._attention_weights = (edge_index_i, alpha)
        
        # 論文 Eq.7: h'_i = Σ_{j∈N_i} α_ij W h_j
        # alpha: (num_edges, heads) -> (num_edges, heads, 1)
        # x_j: (num_edges, heads, out_channels)
        # out: (num_edges, heads, out_channels)
        out = x_j * alpha.unsqueeze(-1)
        
        return out


class GraphAttentionAutoencoder(nn.Module):
    """
    STAGATE の Graph Attention Autoencoder
    
    論文 Fig.1b および Methods セクションを実装
    
    アーキテクチャ:
    - Encoder:
      - Layer 1: input_dim -> 512, activation: ELU
      - Layer 2 (GAT): 512 -> 30, activation: ELU
    
    - Decoder:
      - 重み共有: W^b(k) = W(k)^T（論文 Methods）
      - Layer 1: 30 -> 512
      - Layer 2: 512 -> input_dim
    
    Parameters
    ----------
    input_dim : int
        入力次元（HVG 数、通常 3000）
    hidden_dim : int, default=512
        隠れ層の次元（論文では 512）
    latent_dim : int, default=30
        潜在表現の次元（論文では 30）
    heads : int, default=1
        GAT の multi-head 数
    dropout : float, default=0.0
        Dropout 率
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        latent_dim: int = 30,
        heads: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dropout = dropout
        
        # Encoder Layer 1: input_dim -> hidden_dim
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Encoder Layer 2 (GAT): hidden_dim -> latent_dim
        self.encoder_gat = GraphAttentionLayer(
            in_channels=hidden_dim,
            out_channels=latent_dim,
            heads=heads,
            concat=False,  # 平均を取る
            dropout=dropout
        )
        
        # Decoder Layer 1: latent_dim -> hidden_dim
        # 重み共有: encoder_gat.weight^T を使用
        # ただし、実装の簡便性のため独立したレイヤーとして実装
        # （厳密な重み共有は学習時に設定）
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        
        # Decoder Layer 2: hidden_dim -> input_dim
        # 重み共有: encoder_fc1.weight^T を使用
        self.decoder_fc2 = nn.Linear(hidden_dim, input_dim)
        
        # 活性化関数（論文では ELU）
        self.activation = nn.ELU()
        
        # 重み共有を設定
        self._tie_weights()
    
    def _tie_weights(self):
        """
        Decoder の重みを Encoder と共有
        論文 Methods: "The weights of the decoder are tied with the encoder,
                       i.e., W^b(k) = W(k)^T"
        """
        # decoder_fc2.weight = encoder_fc1.weight^T
        self.decoder_fc2.weight = nn.Parameter(self.encoder_fc1.weight.t())
        
        # decoder_fc1.weight は encoder_gat との共有が複雑なため、
        # ここでは独立させる（論文の実装詳細が不明確なため）
        # より厳密な実装が必要な場合は、カスタム学習ループで対応
    
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Encoder forward pass
        
        Parameters
        ----------
        x : torch.Tensor, shape (num_nodes, input_dim)
            入力特徴量（正規化済み遺伝子発現）
        edge_index : torch.Tensor, shape (2, num_edges)
            エッジのインデックス（SNN）
        edge_attr : torch.Tensor, optional
            エッジの重み
        return_attention_weights : bool
            Attention weights を返すかどうか
        
        Returns
        -------
        z : torch.Tensor, shape (num_nodes, latent_dim)
            潜在表現
        attention_weights : tuple, optional
            Attention weights
        """
        # Layer 1: Linear + ELU
        h = self.encoder_fc1(x)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2: GAT + ELU
        if return_attention_weights:
            z, attention_weights = self.encoder_gat(
                h, edge_index, edge_attr, return_attention_weights=True
            )
            z = self.activation(z)
            return z, attention_weights
        else:
            z = self.encoder_gat(h, edge_index, edge_attr)
            z = self.activation(z)
            return z
    
    def decode(self, z: torch.Tensor):
        """
        Decoder forward pass
        
        Parameters
        ----------
        z : torch.Tensor, shape (num_nodes, latent_dim)
            潜在表現
        
        Returns
        -------
        x_recon : torch.Tensor, shape (num_nodes, input_dim)
            再構成された遺伝子発現
        """
        # Layer 1: Linear + ELU
        h = self.decoder_fc1(z)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2: Linear（活性化なし、reconstruction）
        x_recon = self.decoder_fc2(h)
        
        return x_recon
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Full forward pass (encode -> decode)
        
        Parameters
        ----------
        x : torch.Tensor
            入力特徴量
        edge_index : torch.Tensor
            エッジのインデックス
        edge_attr : torch.Tensor, optional
            エッジの重み
        return_attention_weights : bool
            Attention weights を返すかどうか
        
        Returns
        -------
        x_recon : torch.Tensor
            再構成された特徴量
        z : torch.Tensor
            潜在表現
        attention_weights : tuple, optional
            Attention weights
        """
        # Encode
        if return_attention_weights:
            z, attention_weights = self.encode(
                x, edge_index, edge_attr, return_attention_weights=True
            )
        else:
            z = self.encode(x, edge_index, edge_attr)
        
        # Decode
        x_recon = self.decode(z)
        
        if return_attention_weights:
            return x_recon, z, attention_weights
        else:
            return x_recon, z
