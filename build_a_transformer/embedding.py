# Embedding module

import torch

import conf


def build_emb_model() -> torch.nn.Module:
    return torch.nn.Embedding(
        num_embeddings=conf.VOCAB_SIZE, embedding_dim=conf.EMBEDDING_DIM
    )
