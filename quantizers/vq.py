
import torch
from torch import nn
from torch.nn import functional as F

from torch import distributed as dist

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain('tanh'))
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, continuous_relax=False, temperature=1., hard=False):
        input = input.permute(0, 2, 3, 1)
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        ) # dist map, shape=[*, n_embed]

        if not continuous_relax:
            # argmax + lookup
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            # print(embed_ind.shape)
            # print(input.shape)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            # gumbel softmax weighted sum
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            # gumbel softmax hard lookup
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)

        if self.training and ((continuous_relax and hard) or (not continuous_relax)):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            # maybe need replace a KL term here
            qy = (-dist).softmax(-1)
            diff = torch.sum(qy * torch.log(qy * self.n_embed + 1e-20), dim=-1).mean() # KL
            #diff = (quantize - input).pow(2).mean().detach() # gumbel softmax do not need diff
            quantize = quantize.to(memory_format=torch.channels_last)
        quantize = quantize.permute(0, 3, 1, 2)
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
class VectorQuantizeEMA(nn.Module):
    def __init__(self,
                    args,
                    embedding_dim,
                    n_embed,
                    commitment_cost=1,
                    decay=0.99,
                    eps=1e-5):
        super().__init__()
        self.args = args
        self.ema = True if args.quantizer == 'ema' else False

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        
        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1. / n_embed, 1. / n_embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.embed.weight.data.clone())
        
        self.decay = decay
        self.eps = eps
        
    def forward(self, z_e):
        B, C, H, W = z_e.shape
        
        # z_e = z
        z_e = z_e.permute(0, 2, 3, 1) # (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim) # (B*H*W, C)

        dist = (
            flatten.pow(2).sum(1, keepdim=True) # (B*H*W, 1)
            - 2 * flatten @ self.embed.weight.t() # (B*H*W, n_embed)
            + self.embed.weight.pow(2).sum(1, keepdim=True).t() # (1, n_embed)
        )#(B*H*W, n_embed)
        _, embed_ind = (-dist).max(1) # choose the nearest neighboor
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # (BHW, n_embed)
        embed_ind = embed_ind.view(B, H, W) # 

        z_q = self.embed_code(embed_ind) # B, H, W, C
        
        if self.training and self.ema:
            embed_onehot_sum = embed_onehot.sum(0) # 
            embed_sum = (flatten.transpose(0, 1) @ embed_onehot).transpose(0, 1) # 
            
            all_reduce(embed_onehot_sum.contiguous())
            all_reduce(embed_sum.contiguous())
            
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1-self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.weight.data.copy_(embed_normalized)
        
        if self.ema:
            diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean()
        else:
            diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean() \
                    + (z_q - z_e.detach()).pow(2).mean()

        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2) # B,H,W,C -> B,C,H,W
        return z_q, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)

if __name__ == '__main__':
    pass