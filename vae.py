import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from utils import prepare_vae_data
import models


def run_epoch(dataloader, model, optimizer, beta, device, train):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss, epoch_mse, epoch_kld = 0., 0., 0.
    for x in dataloader:
        optimizer.zero_grad()

        x = x.to(device)

        y, mu, log_var = model(x)
        mse, kld = model.calculate_loss(x, y, mu, log_var)

        loss = mse + beta * kld

        epoch_loss += loss.item()
        epoch_mse += mse.item()
        epoch_kld += kld.item()

        if train:
            loss.backward()
            optimizer.step()

    epoch_loss /= len(dataloader)
    epoch_mse /= len(dataloader)
    epoch_kld /= len(dataloader)

    return epoch_loss, epoch_mse, epoch_kld


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, required=True)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-hd", "--hidden_dim", type=int, default=128)
    parser.add_argument("-ld", "--latent_dim", type=int, default=64)
    parser.add_argument("-nl", "--num_layers", type=int, default=2)
    parser.add_argument("-do", "--dropout", type=float, default=0.25)
    parser.add_argument("-b", "--beta", type=float, default=0.25)
    parser.add_argument("-m", "--model_state_dict", type=str)
    args = parser.parse_args()

    data = pd.read_csv(args.data, index_col=[0, 1])
    
    model = models.VAE(input_dim=data.shape[1],
                        hidden_dim=args.hidden_dim,
                        latent_dim=args.latent_dim,
                        num_layers=args.num_layers,
                        dropout=args.dropout).to(device)
    print(model)

    if not args.model_state_dict:
        dataloader_train, dataloader_test, _ = prepare_vae_data(data.values, split=True, batch_size=args.batch_size)    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        os.makedirs("log/models", exist_ok=True)
        for epoch in range(1, 101):
            s = time.time()
            loss_train, mse_train, kld_train = run_epoch(dataloader_train, model, optimizer, args.beta, device, True)
            loss_test, mse_test, kld_test = run_epoch(dataloader_test, model, optimizer, args.beta, device, False)
            e = time.time()

            t = f"epoch: {epoch:04d} " + \
                f"train loss: {loss_train:.3f}/{mse_train:.3f}/{kld_train:.3f} " + \
                f"test loss: {loss_test:.3f}/{mse_test:.3f}/{kld_test:.3f} " + \
                f"time elapsed: {int(e - s)} sec."
            print(t)

            if epoch % 10 == 0:
                torch.save(model.state_dict(), f"log/models/model_vae_{epoch}.pth")
    else:
        index = data.index

        dataloader, _ = prepare_vae_data(data.values, split=False, batch_size=args.batch_size)
        os.makedirs("log/data", exist_ok=True)

        model.load_state_dict(torch.load(args.model_state_dict))
        model.eval()

        zs = [model(x.to(device), use_mu=True)[1].cpu().detach().numpy() for x in dataloader]
        zs = pd.DataFrame(np.vstack(zs), index=index)
        zs = zs.reindex(index)
        idx = data.notna().sum(axis=1) == 0
        zs.loc[idx, :] = np.nan

        zs.to_csv(f"log/data/{args.data.split('/')[-1].split('.')[0]}_vae.csv")
