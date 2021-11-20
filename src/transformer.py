import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList(
            [LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        layer norm -> self attention -> dropout->add residual 

        Args:
            x ([type]): [description]
            x_k ([type], optional): [description]. Defaults to None.
            x_v ([type], optional): [description]. Defaults to None.
        """

        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        # 一般after就是不会使用layer norm
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after & self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    # triu function return the upper triangular part of a matrix of input, the other elements are set to 0
    future_mask = torch.triu(fill_with_neg_inf(
        torch.ones(dim1, dim2)), 1+abs(dim2+dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]

def fill_with_neg_inf(t):
    """
    FP16-compatible function that fills a tensor with -inf.

    Args:
        t ([type]): [description]

    Returns:
        [type]: [description]
    """
    return t.float().fill_(float('-inf')).type_as(t)

