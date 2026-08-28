"""
Model implementations for GenRec.

This module provides implementations of various recommendation models:

Baseline Models:
    - SASRec: Self-attention with softmax for sequential recommendation
    - HSTU: SiLU attention + update gate + temporal bias

Generative Models:
    - RQVAE: Vector quantized VAE for semantic ID generation
    - TIGER: Generative retrieval with trie-based constrained decoding
    - LCRec: LLM with collaborative semantics via codebook tokens
    - COBRA: Sparse/Dense hybrid with cascaded representations
    - NoteLLM: Qwen2-based LLM for note recommendation
"""

from genrec.models.rqvae import RqVae, QuantizeForwardMode
from genrec.models.rqkmeans import RqKmeans
from genrec.models.opq_rqkmeans import OpqRqKmeans
from genrec.models.rqopq import RqOpq
from genrec.models.tiger import Tiger
from genrec.models.sasrec import SASRec
from genrec.models.hstu import HSTU
from genrec.models.lcrec import LCRec
from genrec.models.cobra import Cobra

__all__ = [
    "RqVae",
    "QuantizeForwardMode",
    "RqKmeans",
    "OpqRqKmeans",
    "RqOpq",
    "Tiger",
    "SASRec",
    "HSTU",
    "LCRec",
    "Cobra",
]
