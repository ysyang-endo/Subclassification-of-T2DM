import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from utils import prepare_finetune_data
import models


def run_epoch(dataloader, model, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.
    for src, tgt, src_mask, tgt_mask, xm_s, y, y_mask in dataloader:
        optimizer.zero_grad()

        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        xm_s = xm_s.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)

        y_pred, _, _ = model(src, tgt, src_mask, tgt_mask, xm_s)
        loss = model.calculate_mse_loss(y, y_pred, y_mask)

        epoch_loss += loss.item()

        if train:
            loss.backward()
            optimizer.step()

    epoch_loss /= len(dataloader)

    return epoch_loss


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-ds", "--data_s", type=str, required=True)
    parser.add_argument("-dy", "--data_y", type=str, required=True)
    parser.add_argument("-msd", "--model_state_dict", type=str, required=True)
    parser.add_argument("-fmsd", "--finetune_model_state_dict", type=str)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.05)
    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=512)
    parser.add_argument("-fd", "--ff_dim", type=int, default=1024)
    parser.add_argument("-nh", "--num_heads", type=int, default=8)
    parser.add_argument("-nl", "--num_layers", type=int, default=4)
    # hyperparameters for BaseTransformer
    parser.add_argument("-bhd", "--base_hidden_dim", type=int, default=512)
    parser.add_argument("-bfd", "--base_ff_dim", type=int, default=1024)
    parser.add_argument("-bnh", "--base_num_heads", type=int, default=8)
    parser.add_argument("-bnl", "--base_num_layers", type=int, default=4)
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col=[0, 1])
    index = data.index.get_level_values(0).unique()
    data_s = pd.read_csv(args.data_s, index_col=[0, 1])
    data_y = pd.read_csv(args.data_y, index_col=[0, 1])
    mask = data.notna()
    mask_s = data_s.notna()
    mask_y = data_y.notna()

    mask = [v.values for _, v in mask.groupby(level=0)]
    data = [v.values for _, v in data.groupby(level=0)]
    mask_s = [v.values for _, v in mask_s.groupby(level=0)]
    data_s = [v.values for _, v in data_s.groupby(level=0)]
    mask_y = [v.values for _, v in mask_y.groupby(level=0)]
    data_y = [v.values for _, v in data_y.groupby(level=0)]

    model = models.FinetuneTransformer(input_dim=data[0].shape[1],
                                       hidden_dim=args.hidden_dim,
                                       num_heads=args.num_heads,
                                       ff_dim=args.ff_dim,
                                       num_layers=args.num_layers,
                                       model_state_dict=args.model_state_dict,
                                       input_dim_s=int(data_s[0].shape[1] * 2),
                                       device=device).to(device)
    print(model)

    if not args.finetune_model_state_dict:
        dataloader_train, dataloader_test, _ = prepare_finetune_data(data, mask, data_s, mask_s, data_y, mask_y, split=True, batch_size=args.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=args.learning_rate,
                                        weight_decay=args.weight_decay)
        
        os.makedirs("log/models", exist_ok=True)
        for epoch in range(1, 301):
            s = time.time()
            loss_train = run_epoch(dataloader_train, model, optimizer, device, True)
            loss_test = run_epoch(dataloader_test, model, optimizer, device, True)
            e = time.time()

            t = f"epoch: {epoch:04d} " + \
                f"train loss: {loss_train:.5f} " + \
                f"test loss: {loss_test:.5f} " + \
                f"time elapsed: {int(e - s)} sec."
            print(t)

            if epoch % 50 == 0:
                torch.save(model.state_dict(), f"log/models/model_finetune_{epoch}.pth")
    else:
        dataloader, _ = prepare_finetune_data(data, mask, data_s, mask_s, data_y, mask_y, split=False, batch_size=args.batch_size)
        os.makedirs("log/data", exist_ok=True)

        model.load_state_dict(torch.load(args.finetune_model_state_dict))
        model.eval()

        zs = []
        for src, tgt, src_mask, tgt_mask, xm_s, y, y_mask in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            xm_s = xm_s.to(device)
        
            y_pred, z, _ = model(src, tgt, src_mask, tgt_mask, xm_s)

            z = z.cpu().detach().numpy()
            z = z.reshape(z.shape[0], -1)
            zs.append(z)

        zs = pd.DataFrame(np.vstack(zs), index=index)
        zs = zs.to_csv(f"log/data/{args.data.split('/')[-1].split('.')[0]}_embedding.csv")