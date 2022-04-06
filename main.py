import argparse
import logging
from pprint import pprint

import torch
import tqdm
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

from synthetic_dataset import get_general_vae_dataset
from model import LinearBetaVAE

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=512)

parser.add_argument('--input_dim', default=5)
parser.add_argument('--target_dim', default=5)
parser.add_argument('--latent_dim', default=5)
parser.add_argument('--eta_dec_sq', default=1)
parser.add_argument('--eta_prior_sq', default=1)

parser.add_argument('--batch_size', default=16)
parser.add_argument('--epoch', default=100)
parser.add_argument('--lr', default=1e-2)

logging.basicConfig(filename='linear_beta_vae.log', filemode='wt', level=logging.INFO)

def train(dataset, model: nn.Module, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    trajectory = []
    with tqdm.trange(args.epoch) as t:
        for e in t:
            total_loss = 0
            total_rec_loss = 0
            total_kl_loss = 0

            for i, (x, y) in enumerate(loader):
                optimizer.zero_grad()
                fetched = model(x, y)
                loss = fetched['loss']
                total_loss += loss.item()
                total_rec_loss += fetched['rec_loss'].item()
                total_kl_loss += fetched['kl_loss'].item()
                loss.backward()
                optimizer.step()

            L = len(loader)
            traj = {'epoch': e,
                    'total': total_loss/L, 
                    'rec': total_rec_loss/L, 
                    'kl': total_kl_loss/L}
            logging.info(traj)
            trajectory.append(traj)
            
    return trajectory


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = get_general_vae_dataset(
        num_samples=args.num_samples,
        input_dim=args.input_dim,
        target_dim=args.target_dim,
        singular_values=[1,1,1,1,1]
    )

    beta_list = list(i / 10 for i in range(25))
    # beta_list = [150, 10, 0.001]

    total_loss = []
    rec_loss = []
    kl_loss = []
    final_traj = []
    for beta in beta_list:
        print('beta = ', beta)

        model = LinearBetaVAE(input_dim=args.input_dim,
                              latent_dim=args.latent_dim,
                              target_dim=args.target_dim,
                              eta_dec_sq=args.eta_dec_sq,
                              eta_prior_sq=args.eta_prior_sq,
                              beta=beta)

        traj = train(dataset, model, args)
        total_loss.append(traj[-1]['total'])
        rec_loss.append(traj[-1]['rec'])
        kl_loss.append(traj[-1]['kl'])

        final_traj.append(traj[-1])

    pprint(list(zip(beta_list, final_traj)))

    plt.scatter(beta_list, total_loss, label='total')
    plt.scatter(beta_list, rec_loss, label='rec')
    plt.scatter(beta_list, kl_loss, label='kl')
    plt.legend()
    plt.title('loss with respect to beta')
    plt.xlabel('beta')
    plt.ylabel('loss')
    plt.savefig('linear_vae_beta_loss.png')