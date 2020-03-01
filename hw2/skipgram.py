import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class SkipGram(nn.Module):
    def __init__(self, vocab_size, emb_dimension):
        """

        Args:
            vocab_size (int): Size of entire vocabulary
            emb_dimension (int): Dimensionality of embeddings
        Returns:

        """
        super(SkipGram, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.u_emb = nn.Embedding(vocab_size, emb_dimension, sparse=True)
        self.v_emb = nn.Embedding(vocab_size, emb_dimension, sparse=True)

        # initialize embeddings with Xavier initialization
        init_range = (2.0 / (vocab_size + emb_dimension)) ** 0.5
        self.u_emb.weight.data.uniform_(-init_range, init_range)
        self.v_emb.weight.data.uniform_(-0,0)

    def forward(self, pos_u, pos_v, neg_v):
        """ Forward pass
        Args:
            pos_u (tensor): Ids of center words of positive word pairs of size batch_size
            pos_v (tensor): Ids of neighbor words of positive word pairs of size batch_size
            neg_v (tensor): Ids of center words of negative word pairs of size batch_size
        Returns:
            Loss obtained after forward pass

        """
        # dim: [batch_size, emb_dimension]
        u = self.u_emb(pos_u)
        # dim: [batch_size, emb_dimension]
        v = self.v_emb(pos_v)
        # dim: [batch_size, neg_v size, emb_dimension]
        n_v = self.v_emb(neg_v)

        # elementwise multiplication across batches
        score = torch.mul(u, v)
        score = torch.sum(score, dim=1)
        # dim score: [batch_size]
        score = F.logsigmoid(score).squeeze()

        # batchwise matrix multiplication bmm
        # n_v * u, dim: [batch_size, neg_v_size, emb_dimension] * [batch_size, emb_dimension, 1]
        # = [batch_size, neg_v_size, 1] -> unsqueeze
        neg_score = torch.bmm(n_v, u.unsqueeze(2)).squeeze()
        neg_score = -1 * torch.sum(neg_score, dim=1)
        # dim neg_score: [batch_size, 1] -> squeeze -> [batch_size]
        neg_score = F.logsigmoid(neg_score).squeeze()

        return -torch.mean(score + neg_score)

    def map_embedding(self, inputs):
        """ Predicts embedding
        Args:
            inputs:

        Returns:
        """

        return self.u_emb(inputs)