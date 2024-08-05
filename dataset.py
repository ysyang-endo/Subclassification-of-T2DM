import numpy as np
import torch
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(self, x):
        x = torch.from_numpy(x)
        m = ~torch.isnan(x)
        x = torch.nan_to_num(x, 0)
        
        self.x = torch.cat([x, m], axis=1).float()
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class PretrainDataset(Dataset):
    def __init__(self, x, m, max_length=68):
        triu_mask = torch.triu(torch.ones(max_length, max_length), diagonal=0).bool()

        self.src_mask, self.tgt_mask = [], []
        for m_ in m:
            m_ = torch.from_numpy(m_).sum(axis=1) == 0
            m_ = torch.cat([m_, torch.ones(max_length - m_.shape[0])])
            m_ = m_.unsqueeze(0).repeat(max_length, 1).bool()
            self.src_mask.append(m_)

            m_ = torch.cat([torch.zeros(max_length, 1), m_[:, :-1]], axis=1).bool()
            m_ = m_ | triu_mask
            self.tgt_mask.append(m_)

        self.src, self.tgt = [], []
        for x_ in x:
            x_ = torch.from_numpy(x_)
            x_ = torch.nan_to_num(x_, 0)
            x_ = torch.cat([x_, torch.zeros(max_length - x_.shape[0], x_.shape[1])], dim=0)
            self.src.append(x_.float())
            self.tgt.append(x_[:-1, :].float()) # SOS embedding in model

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_mask[idx], self.tgt_mask[idx]


class FinetuneDataset(Dataset):
    def __init__(self, x, m, x_s, m_s, y, m_y, max_length=68):
        triu_mask = torch.triu(torch.ones(max_length, max_length), diagonal=0).bool()

        self.src_mask, self.tgt_mask = [], []
        for m_ in m:
            m_ = torch.from_numpy(m_).sum(axis=1) == 0
            m_ = torch.cat([m_, torch.ones(max_length - m_.shape[0])])
            m_ = m_.unsqueeze(0).repeat(max_length, 1).bool()
            self.src_mask.append(m_)

            m_ = torch.cat([torch.zeros(max_length, 1), m_[:, :-1]], axis=1).bool()
            m_ = m_ | triu_mask
            self.tgt_mask.append(m_)

        self.src, self.tgt = [], []
        for x_ in x:
            x_ = torch.from_numpy(x_)
            x_ = torch.nan_to_num(x_, 0)
            x_ = torch.cat([x_, torch.zeros(max_length - x_.shape[0], x_.shape[1])], dim=0)
            self.src.append(x_.float())
            self.tgt.append(x_[:-1, :].float()) # SOS embedding in model

        self.xm_s = []
        for x_s_, m_s_ in zip(x_s, m_s):
            x_s_ = torch.from_numpy(x_s_)
            m_s_ = torch.from_numpy(m_s_)
            x_s_ = torch.nan_to_num(x_s_, 0)
            xm_s_ = torch.cat([x_s_, m_s_], axis=1)
            xm_s_ = torch.cat([xm_s_, torch.zeros([max_length - xm_s_.size(0), xm_s_.size(1)])], axis=0)
            self.xm_s.append(xm_s_.float())

        self.y, self.y_mask = [], []
        for y_, m_y_ in zip(y, m_y):
            y_ = torch.from_numpy(y_)
            m_y_ = torch.from_numpy(m_y_)
            y_ = torch.nan_to_num(y_, 0)
            m_y_ = torch.cat([m_y_, torch.zeros([max_length - m_y_.size(0), m_y_.size(1)])], axis=0)
            y_ = torch.cat([y_, torch.zeros([max_length - y_.size(0), y_.size(1)])], axis=0)
            self.y.append(y_.float())
            self.y_mask.append(m_y_)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], \
               self.src_mask[idx], self.tgt_mask[idx], \
               self.xm_s[idx], \
               self.y[idx], self.y_mask[idx]
