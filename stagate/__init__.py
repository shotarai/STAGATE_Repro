"""
STAGATE: Spatial Transcriptomics Analysis with Graph Attention auto-encoder
論文完全再現実装
"""

from .preprocessing import preprocess_data
from .snn import build_spatial_network
from .celltype_snn import build_celltype_aware_snn
from .model import GraphAttentionAutoencoder
from .train import train_stagate
from .clustering import cluster_domains
from .visualization import (
    plot_spatial_domains,
    plot_umap,
    plot_attention_weights
)
from .utils import set_seed

__version__ = "1.0.0"
__author__ = "STAGATE Reproduction Team"

__all__ = [
    "preprocess_data",
    "build_spatial_network",
    "build_celltype_aware_snn",
    "GraphAttentionAutoencoder",
    "train_stagate",
    "cluster_domains",
    "plot_spatial_domains",
    "plot_umap",
    "plot_attention_weights",
    "set_seed",
]
