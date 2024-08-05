import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout=0.25):
        super(VAE, self).__init__()

        encoder_layers = []
        for i in range(num_layers):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dim*2, hidden_dim))
            elif i == num_layers - 1:
                encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
            else:
                encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))

            if i < num_layers - 1:
                encoder_layers.append(nn.GELU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)


        decoder_layers = [nn.Dropout(p=dropout)]
        for i in range(num_layers):
            if i == 0:
                decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
            elif i == num_layers - 1:
                decoder_layers.append(nn.Linear(hidden_dim, input_dim))
            else:
                decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                decoder_layers.append(nn.GELU())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, use_mu=False):
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        if use_mu:
            z = mu
        else:
            z = self.reparameterize(mu, log_var)

        y = self.decoder(z)
        return y, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def calculate_loss(self, x, y, mu, log_var):
        m = x[:, x.size(1)//2:]
        x = x[:, :x.size(1)//2]
        
        recon = F.mse_loss(y, x, reduction="none")
        recon = recon * m
        recon = recon.sum(axis=0) / (m.sum(axis=0) + 1e-8)
        recon = recon.mean()

        kld = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), axis=1)
        kld = torch.mean(kld)

        return recon, kld


class BaseTransformer(nn.Module):
    def __init__(self, input_dim,
                       hidden_dim,
                       num_heads,
                       ff_dim,
                       num_layers,
                       device):
        super(BaseTransformer, self).__init__()
        self.embedding = Embedding(input_dim, hidden_dim, device)
        
        self.encoder_layers = nn.ModuleList([Encoder(hidden_dim,
                                                     num_heads,
                                                     ff_dim) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(hidden_dim,
                                                     num_heads,
                                                     ff_dim) for _ in range(num_layers)])
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.embedding(src)
        tgt = self.embedding(tgt, add_sos=True)

        attns1 = []
        for layer in self.encoder_layers:
            src, attn1 = layer(src, src_mask)
            attns1.append(attn1)

        mem = src.clone()
                
        attns2, attns3 = [], []
        for layer in self.decoder_layers:
            tgt, attn2, attn3 = layer(tgt, tgt_mask, mem, src_mask)
            attns2.append(attn2)
            attns3.append(attn3)

        x = self.linear(tgt)
        return x, mem, tgt, (attns1, attns2, attns3)

    def calculate_mse_loss(self, x, y, mask_indices):
        x = x.view(-1, x.size(2))[mask_indices, :]
        y = y.view(-1, y.size(2))[mask_indices, :]
        loss = F.mse_loss(x, y)
        return loss
    
    def calculate_bce_loss(self, x, y, mask_indices, weight):
        x = x[:, :, :-1]
        y = y[:, :, :-1]

        x = x.view(-1, x.size(2))[mask_indices, :]
        y = y.view(-1, y.size(2))[mask_indices, :]
        loss = F.binary_cross_entropy_with_logits(y, x, weight=weight)
        return loss


class FinetuneTransformer(nn.Module):
    def __init__(self, input_dim,
                       hidden_dim,
                       num_heads,
                       ff_dim,
                       num_layers,
                       model_state_dict,
                       input_dim_s,
                       device):
        super(FinetuneTransformer, self).__init__()
        self.tf = BaseTransformer(input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_heads=num_heads,
                                  ff_dim=ff_dim,
                                  num_layers=num_layers,
                                  device=device)
        self.tf.load_state_dict(torch.load(model_state_dict))
        for param in self.tf.parameters():
            param.requires_grad = False

        self.tgt_mixer = nn.Linear(hidden_dim+input_dim_s, hidden_dim)
        self.embedding = Embedding(hidden_dim, hidden_dim, device)

        self.decoder = nn.ModuleList([Decoder(hidden_dim,
                                              num_heads,
                                              ff_dim)
                                      for _ in range(num_layers)])

        self.linear1 = nn.Linear(hidden_dim, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, src,
                      tgt,
                      src_mask,
                      tgt_mask,
                      x_s):
        _, mem, tgt, _ = self.tf(src, tgt, src_mask, tgt_mask)
        tgt = torch.cat([tgt, x_s], axis=-1)
        tgt = self.tgt_mixer(tgt)
        
        tgt = self.embedding(tgt)
        
        attns1, attns2 = [], []
        for layer in self.decoder:
            tgt, attn1, attn2 = layer(tgt, tgt_mask, mem, src_mask)
            attns1.append(attn1)
            attns2.append(attn2)

        x = F.gelu(tgt)
        z = self.linear1(x)
        x = F.gelu(z)
        x = self.linear2(x)

        return x, z, (attns1, attns2)

    def calculate_mse_loss(self, y, y_pred, y_mask):
        loss = F.mse_loss(y, y_pred, reduction="none")
        loss = loss * y_mask
        loss = loss.sum() / y_mask.sum()
        return loss
    
    def calculate_bce_loss(self, y, y_pred, eps=1e-16):
        loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y)
        return loss


class Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Embedding, self).__init__()
        self.emb = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, device)
        self.norm = nn.LayerNorm(hidden_dim)
        self.sos = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, add_sos=False):
        x = self.emb(x)

        if add_sos:
            batch_size = x.size(0)            
            start_token = self.sos.repeat(batch_size, 1, 1)
            x = torch.cat([start_token, x], dim=1)

        x = self.pos_enc(x)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, device, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.enc = torch.zeros(max_len, hidden_dim, device=device)
        self.enc.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, hidden_dim, step=2, device=device).float()

        self.enc[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_dim)))
        self.enc[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_dim)))

    def forward(self, x):
        seq_len = x.size(1)
        enc = self.enc[:seq_len, :]
        x = x + enc
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super(Encoder, self).__init__()
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FeedForward(hidden_dim, ff_dim)

    def forward(self, src, m_src):
        src, attn = self.attn(src, m_src)
        src = self.ffn(src)
        return src, attn


class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super(Decoder, self).__init__()
        self.masked_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = FeedForward(hidden_dim, ff_dim)

    def forward(self, x_dec, m_dec, mem, m_mem):
        x_dec, masked_attn = self.masked_attn(x_dec, m_dec)
        x_dec, attn = self.attn(x_dec, m_mem, mem=mem)
        x_dec = self.ffn(x_dec)
        return x_dec, masked_attn, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask, mem=None):
        batch_size, seq_len, hidden_dim = x.size()

        x_ = x

        if mem is not None:
            q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.w_k(mem).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.w_v(mem).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn = attn.masked_fill_(mask, -1e9)
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = self.norm(x + x_)
        return x, attn


class FeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x_ = x

        x = F.gelu(self.linear1(x))
        x = self.linear2(x)

        x = self.norm(x + x_)
        return x
