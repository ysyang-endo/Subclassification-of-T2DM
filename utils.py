import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import VAEDataset, PretrainDataset, FinetuneDataset


def prepare_vae_data(data, split, batch_size):
    if split:
        data_train, data_test = train_test_split(data, test_size=.1, random_state=42)

        scaler = StandardScaler()
        data_train = scaler.fit_transform(data_train)
        data_test = scaler.transform(data_test)

        dataset_train = VAEDataset(data_train)
        dataset_test = VAEDataset(data_test)

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=32)
        dataloader_test = DataLoader(dataset_test,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=32)
        return dataloader_train, dataloader_test, scaler
    else:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        
        dataset = VAEDataset(data)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=32)
        return dataloader, scaler



def pretrain_cfn(batch, mlm_ratio=0.15, mask_ratio=0.80, same_ratio=0.10, rnd_ratio=0.10):
    src, tgt, src_mask, tgt_mask = zip(*batch)

    src = torch.stack(src)
    tgt = torch.stack(tgt)
    src_mask = torch.stack(src_mask)
    tgt_mask = torch.stack(tgt_mask)
    x = src.clone()

    flat_mask = torch.diagonal(src_mask, dim1=-2, dim2=-1).reshape(-1)
    non_padded_indices = torch.nonzero(flat_mask == 0, as_tuple=False).squeeze()

    num_mlm = int(mlm_ratio * len(non_padded_indices))
    masked_indices = non_padded_indices[torch.randperm(len(non_padded_indices))[:num_mlm]]
    
    non_padded_embeddings = src.view(-1, src.size(-1))[non_padded_indices]
    non_padded_embeddings = non_padded_embeddings[torch.randperm(len(non_padded_embeddings))]

    mask_decision = torch.rand(num_mlm)
    mask_type_indices = {
        "mask": masked_indices[mask_decision < mask_ratio],
        "random": masked_indices[mask_decision >= mask_ratio + same_ratio]
    }

    src.view(-1, src.shape[-1])[mask_type_indices["mask"]] = 0
    src.view(-1, src.shape[-1])[mask_type_indices["random"]] = non_padded_embeddings[:len(mask_type_indices["random"])]

    return src, tgt, src_mask, tgt_mask, x, masked_indices


def prepare_pretrain_data(data, mask, batch_size):
    scaler = StandardScaler()
    scaler.fit(np.vstack(data))
    data = [scaler.transform(d) for d in data]

    dataset = PretrainDataset(data, mask)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=pretrain_cfn,
                            num_workers=32)
    return dataloader, scaler


def prepare_finetune_data(data, mask, data_s, mask_s, data_y, mask_y, split, batch_size):
    if split:
        idx = list(range(len(data)))
        idx_train, idx_test = train_test_split(idx, test_size=.1, random_state=42)
        
        data_train = [data[i] for i in idx_train]
        data_test = [data[i] for i in idx_test]
        mask_train = [mask[i] for i in idx_train]
        mask_test = [mask[i] for i in idx_test]
        data_s_train = [data_s[i] for i in idx_train]
        data_s_test = [data_s[i] for i in idx_test]
        mask_s_train = [mask_s[i] for i in idx_train]
        mask_s_test = [mask_s[i] for i in idx_test]
        data_y_train = [data_y[i] for i in idx_train]
        data_y_test = [data_y[i] for i in idx_test]
        mask_y_train = [mask_y[i] for i in idx_train]
        mask_y_test = [mask_y[i] for i in idx_test]    
       
        scaler_d = StandardScaler()
        scaler_d = scaler_d.fit(np.vstack(data_train))
        data_train = [scaler_d.transform(d) for d in data_train]
        data_test = [scaler_d.transform(d) for d in data_test]

        scaler_s = StandardScaler()
        scaler_s = scaler_s.fit(np.vstack(data_s_train))
        data_s_train = [scaler_s.transform(d) for d in data_s_train]
        data_s_test = [scaler_s.transform(d) for d in data_s_test]

        scaler_y = StandardScaler()
        scaler_y = scaler_y.fit(np.vstack(data_y_train))
        data_y_train = [scaler_y.transform(d) for d in data_y_train]
        data_y_test = [scaler_y.transform(d) for d in data_y_test]

        dataset_train = FinetuneDataset(data_train,
                                        mask_train,
                                        data_s_train,
                                        mask_s_train,
                                        data_y_train,
                                        mask_y_train)
        dataset_test = FinetuneDataset(data_test,
                                       mask_test,
                                       data_s_test,
                                       mask_s_test,
                                       data_y_test,
                                       mask_y_test)

        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0)
        dataloader_test = DataLoader(dataset_test,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=0)
        return dataloader_train, dataloader_test, (scaler_d, scaler_s, scaler_y)
    else:
        scaler_d = StandardScaler()
        scaler_d = scaler_d.fit(np.vstack(data))
        data = [scaler_d.transform(d) for d in data]

        scaler_s = StandardScaler()
        scaler_s = scaler_s.fit(np.vstack(data_s))
        data_s = [scaler_s.transform(d) for d in data_s]

        scaler_y = StandardScaler()
        scaler_y = scaler_y.fit(np.vstack(data_y))
        data_y = [scaler_y.transform(d) for d in data_y]
        
        dataset = FinetuneDataset(data,
                                  mask,
                                  data_s,
                                  mask_s,
                                  data_y,
                                  mask_y)

        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0)
        return dataloader, (scaler_d, scaler_s, scaler_y)