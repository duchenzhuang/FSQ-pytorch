from torch import nn
from torch.nn import functional as F
import numpy as np
from quantizers import VectorQuantizeEMA, FSQ, LFQ
import torch

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_channel = args.in_channel
        channel = args.channel
        embed_dim = args.embed_dim

        blocks = [
                    nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                ]

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        in_channel=args.embed_dim
        out_channel=args.in_channel
        channel=args.channel
        
        blocks = [
            nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1),
        ]
        blocks.append(nn.ReLU(inplace=True))
        blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel, channel, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel, out_channel, 1)
                ]
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.args = args
        
        if args.quantizer == 'ema' or args.quantizer == 'origin':
            self.quantize_t = VectorQuantizeEMA(args, args.embed_dim, args.n_embed)

        elif args.quantizer == 'lfq':
            self.quantize_t = LFQ(codebook_size = 2**args.lfq_dim, dim = args.lfq_dim, entropy_loss_weight=args.entropy_loss_weight, commitment_loss_weight=args.codebook_loss_weight)
            # args.embed_dim = args.lfq_dim
        elif args.quantizer == 'fsq':
            self.quantize_t = FSQ(levels=args.levels)
            # args.embed_dim = len(args.levels)
        else:
            print('quantizer error!')
            exit()

        self.enc = Encoder(args)
        self.dec = Decoder(args)
         
        
        
    def forward(self, input):
        quant_t, diff, _, = self.encode(input)
        dec = self.dec(quant_t)
        return dec, diff

    def encode(self, input):
        logits = self.enc(input)
        if self.args.quantizer == 'ema'  or self.args.quantizer == 'origin':
            quant_t, diff_t, id_t = self.quantize_t(logits)
            # quant_t = quant_t.permute(0, 3, 1, 2) have change the dimension in quantizer
            diff_t = diff_t.unsqueeze(0)
        
        elif self.args.quantizer == 'fsq':
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).cuda().float()
        
        elif self.args.quantizer == 'lfq':
            # quantized, indices, entropy_aux_loss = quantizer(image_feats)
            quant_t, id_t, diff_t = self.quantize_t(logits)
        return quant_t, diff_t, id_t

    
    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)

        return dec

