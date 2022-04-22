import argparse
import logging
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from model import LinearBetaVAE, ReLUBetaVAE, TanhBetaVAE
from synthetic_dataset import get_general_vae_dataset, get_synthetic_vae_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=4096)

parser.add_argument('--model', default='linear')
parser.add_argument('--name', type=str, default="")
parser.add_argument('--input_dim', default=5, type=int)
parser.add_argument('--target_dim', default=5, type=int)
parser.add_argument('--latent_dim', default=5, type=int)
parser.add_argument('--hidden_dim', default=8, type=int)
parser.add_argument('--eta_dec_sq', default=1, type=float)
parser.add_argument('--eta_prior_sq', default=1, type=float)

parser.add_argument('--batch_size', default=16)
parser.add_argument('--epoch', default=300)
parser.add_argument('--lr', default=1e-3)


def train(dataset, model: nn.Module, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trajectory = []
    with tqdm.trange(args.epoch) as t:
        for e in t:
            total_loss = 0
            total_rec_loss = 0
            total_kl_loss = 0
            total_enc_norm = 0
            total_dec_norm = 0
            sigma_array_list = []

            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                fetched = model(x, y)
                loss = fetched['loss']
                total_loss += loss.item()
                total_rec_loss += fetched['rec_loss'].item()
                total_kl_loss += fetched['kl_loss'].item()
                sigma_array_list.append(fetched['sigma'].detach().cpu().numpy())
                if args.model == 'linear':
                    total_enc_norm += fetched['enc_norm'].item()
                    total_dec_norm += fetched['dec_norm'].item()
                loss.backward()
                optimizer.step()

            L = len(loader)
            traj = {'epoch': e,
                    'total': total_loss/L,
                    'rec': total_rec_loss/L,
                    'kl': total_kl_loss/L,
                    'enc_norm': total_enc_norm/L,
                    'dec_norm': total_dec_norm/L}
            t.set_postfix(traj)
            logging.info(traj)
            traj['sigma_array'] = np.concatenate(sigma_array_list, axis=0)
            trajectory.append(traj)

    return trajectory

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(filename=f'{args.model}_beta_vae_regression.log', filemode='wt', level=logging.INFO)


    xi_list = [1.2, 1.4, 1.6, 1.8, 2]
    dataset = get_general_vae_dataset(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        target_dim=args.target_dim,
        singular_values=xi_list
    )
    beta_list = list(i for i in range(21))
    # beta_list = [150, 10, 0.001]

    total_loss = []
    rec_loss = []
    kl_loss = []
    sigma_mean_list = []
    sigma_std_list = []
    final_traj = []
    enc_norm = []
    dec_norm = []
    for beta in beta_list:
        print('beta = ', beta)

        if args.model == 'linear':
            VAE = LinearBetaVAE
        elif args.model == 'relu':
            VAE = ReLUBetaVAE
        elif args.model == 'tanh':
            VAE = TanhBetaVAE

        model = VAE(input_dim=args.input_dim,
                    latent_dim=args.latent_dim,
                    target_dim=args.target_dim,
                    hidden_dim=args.hidden_dim,
                    eta_dec_sq=args.eta_dec_sq,
                    eta_prior_sq=args.eta_prior_sq,
                    beta=beta)

        traj = train(dataset, model, args)
        total_loss.append(traj[-1]['total'])
        rec_loss.append(traj[-1]['rec'])
        kl_loss.append(traj[-1]['kl'])

        last_sigma = traj[-1]['sigma_array']

        sigma_mean_list.append(np.mean(last_sigma, axis=0).tolist())
        sigma_std_list.append(np.std(last_sigma, axis=0).tolist())

        # assert len(sigma_mean_list[0]) == len(xi_list)

        if args.model == 'linear':
            enc_norm.append(traj[-1]['enc_norm'])
            dec_norm.append(traj[-1]['dec_norm'])

        final_traj.append(traj[-1])


    data = {'beta': beta_list,
            'total_loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss}

    if args.model == 'linear':
        data['enc_norm'] = enc_norm
        data['dec_norm'] = dec_norm

    for i in range(args.latent_dim):
        data[f'sigma-{i}_mean'] = [sigma_mean[i] for sigma_mean in sigma_mean_list]
        data[f'sigma-{i}_std']  = [sigma_std[i]  for sigma_std in sigma_std_list]

    pd.DataFrame(data).to_csv(f'output/regression_{args.name}{args.model}_losses.csv', index=False)
