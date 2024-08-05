import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from utils import prepare_pretrain_data
import models


def linear_schedule_with_warmup(epoch, warmup_epoch=25, max_epoch=500):
    if epoch < warmup_epoch:
        return float(epoch) / warmup_epoch
    remaining_epochs = max(0, max_epoch - epoch)
    total_decay_epochs = max(1, max_epoch - warmup_epoch)
    return remaining_epochs / total_decay_epochs


def run_epoch(dataloader, model, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.
    for src, tgt, src_mask, tgt_mask, x, masked_indices in dataloader:
        optimizer.zero_grad()

        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        x = x.to(device)
        masked_indices = masked_indices.to(device)

        y, _, _, _ = model(src, tgt, src_mask, tgt_mask)
        loss = model.calculate_mse_loss(x, y, masked_indices)

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
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.05)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=512)
    parser.add_argument("-fd", "--ff_dim", type=int, default=1024)
    parser.add_argument("-nh", "--num_heads", type=int, default=8)
    parser.add_argument("-nl", "--num_layers", type=int, default=4)
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col=[0, 1])
    mask = data.notna()
    mask = [v.values for _, v in mask.groupby(level=0)]
    data = [v.values for _, v in data.groupby(level=0)]

    dataloader, _ = prepare_pretrain_data(data, mask, batch_size=args.batch_size)

    model = models.BaseTransformer(input_dim=data[0].shape[1],
                                   hidden_dim=args.hidden_dim,
                                   num_heads=args.num_heads,
                                   ff_dim=args.ff_dim,
                                   num_layers=args.num_layers,
                                   device=device).to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_schedule_with_warmup)
    
    os.makedirs("log/models", exist_ok=True)
    for epoch in range(1, 501):
        s = time.time()
        loss = run_epoch(dataloader, model, optimizer, device, True)
        e = time.time()

        scheduler.step()

        t = f"epoch: {epoch:04d} " + \
            f"loss: {loss:.5f} " + \
            f"time elapsed: {int(e - s)} sec."
        print(t)

        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"log/models/model_pretrain_{epoch}.pth")
