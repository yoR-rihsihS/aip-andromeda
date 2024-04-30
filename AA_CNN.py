import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np

class AttentionConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, dk, dv, num_heads, kernel_size, padding, height=None, width=None):
        super(AttentionConv2d, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dkh = self.dk // self.num_heads
        self.H = height
        self.W = width

        self.conv_qkv = nn.Conv2d(input_dim, 2*dk + dv, 1)
        self.conv_attn = nn.Conv2d(dv, dv, 1)
        self.conv_out = nn.Conv2d(input_dim, output_dim - dv, kernel_size, padding=padding)
        self.softmax = nn.Softmax(dim=-1)

        self.pe = None
        self.c = 10000

    def compute_pe(self):
        d = self.input_dim
        P = np.zeros((self.H, self.W, self.input_dim))
        for i in range(self.H):
            for j in range(self.W):
                for k in range(int(d/4)):
                    denominator = np.power(self.c, 4*k/d)
                    P[i, j, 2*k] = np.sin(i/denominator)
                    P[i, j, 2*k+1] = np.cos(i/denominator)
                    P[i, j, 2*k+d//2] = np.sin(j/denominator)
                    P[i, j, 2*k+1+d//2] = np.cos(j/denominator)
    
        self.pe = torch.permute(torch.tensor(P, dtype=torch.float32, device='cuda:0'), (2, 0, 1))
        

    def forward(self, input):
        # input = (B x in_channels X h x w)
        conv_out = self.conv_out(input)

        # we have to add positional encodings in the input (B x in_channels X h x w)
        if self.pe is None:
            self.compute_pe()

        input = input + self.pe

        qkv = self.conv_qkv(input)    # batch_size, 2*dk+dv, H, W
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        batch_size, _, H, W = q.size()

        q = q.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        k = k.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        v = v.view([batch_size, self.num_heads, self.dv // self.num_heads, H*W])

        q = q * self.dkh ** -0.5
        logits = einsum('ijkl, ijkm -> ijlm', q, k) # logits.shape = B x Nh x Hw x HW

        weights = self.softmax(logits)
        attn_out = einsum('ijkl, ijfl -> ijfk', weights, v) # attn_out.shape = B x Nh x dvh x HW
        attn_out = attn_out.contiguous().view(batch_size, self.dv, H, W)
        attn_out = self.conv_attn(attn_out)

        output = torch.cat([conv_out, attn_out], dim=1)

        return output


class LinearAttentionConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, dk, dv, num_heads, kernel_size, padding, height=None, width=None):
        super(LinearAttentionConv2d, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dk = dk
        self.dv = dv
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.dkh = self.dk // self.num_heads
        self.H = height
        self.W = width

        self.conv_qkv = nn.Conv2d(input_dim, 2*dk + dv, 1)
        self.conv_attn = nn.Conv2d(dv, dv, 1)
        self.conv_out = nn.Conv2d(input_dim, output_dim - dv, kernel_size, padding=padding)
        self.softmax_row = nn.Softmax(dim=-1)
        self.softmax_col = nn.Softmax(dim=-2)

        self.pe = None
        self.c = 10000

    def compute_pe(self):
        d = self.input_dim
        P = np.zeros((self.H, self.W, self.input_dim))
        for i in range(self.H):
            for j in range(self.W):
                for k in range(int(d/4)):
                    denominator = np.power(self.c, 4*k/d)
                    P[i, j, 2*k] = np.sin(i/denominator)
                    P[i, j, 2*k+1] = np.cos(i/denominator)
                    P[i, j, 2*k+d//2] = np.sin(j/denominator)
                    P[i, j, 2*k+1+d//2] = np.cos(j/denominator)
    
        self.pe = torch.permute(torch.tensor(P, dtype=torch.float32, device='cuda:0'), (2, 0, 1))
        

    def forward(self, input):
        # input = (B x in_channels X h x w)
        conv_out = self.conv_out(input)

        # we have to add positional encodings in the input (B x in_channels X h x w)
        if self.pe is None:
            self.compute_pe()

        input = input + self.pe

        qkv = self.conv_qkv(input) # batch_size, 2*dk+dv, H, W
        q, k, v = torch.split(qkv, [self.dk, self.dk, self.dv], dim=1)
        batch_size, _, H, W = q.size()

        q = q.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        k = k.view([batch_size, self.num_heads, self.dk // self.num_heads, H*W])
        v = v.view([batch_size, self.num_heads, self.dv // self.num_heads, H*W])

        q = q * self.dkh ** -0.5
        k = k * self.dkh ** -0.5
        
        weights_q = self.softmax_row(q) # weights_q.shape = B x Nh x dkh x HW

        weights_k = self.softmax_col(k)
        pre_attn = einsum('ijkl, ijml -> ijkm', weights_k, v) # pre_attn.shape = B x Nh x dkh x dvh
        attn_out = einsum('ijkl, ijkm -> ijml', weights_q, pre_attn)

        attn_out = attn_out.contiguous().view(batch_size, self.dv, H, W)
        attn_out = self.conv_attn(attn_out)

        output = torch.cat([conv_out, attn_out], dim=1)

        return output