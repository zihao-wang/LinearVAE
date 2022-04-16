import matplotlib.pyplot as plt

import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_prefix', type=str)
parser.add_argument('--linear', action='store_true')


def theory_losses(beta, xi_list):
    total_loss = 0
    rec_loss = 0
    kl_loss = 0
    sigma_list = []
    for xi in xi_list:
        if beta < xi ** 2:
            sigma = np.sqrt(beta) / xi
        else:
            sigma = 1
        sigma_list.append(sigma)
        rec_loss += xi ** 2 - max(0, xi - np.sqrt(beta) * sigma) ** 2
        kl_loss  += beta * (sigma ** 2 - 1 - 2 * np.log(sigma + 1e-10))
    
    total_loss = rec_loss + kl_loss
    return total_loss / 2, rec_loss / 2, kl_loss / 2


def plot_linear_beta_VAE_model(args):
    data = pd.read_csv(args.input_file)
    plt.figure()

    plt.scatter(data['beta'], data['total_loss'], marker='s', label='total')
    plt.scatter(data['beta'], data['rec_loss'], marker='d', label='rec')
    plt.scatter(data['beta'], data['kl_loss'], marker='D', label=r'$\beta$ KL')

    beta_list = data['beta'].tolist()
    xi_list = [1, 2, 3, 4, 5]

    theory_total_loss, theory_rec_loss, theory_kl_loss = [], [], []
    for beta in beta_list:
        t, r, k = theory_losses(beta, xi_list)
        theory_total_loss.append(t)
        theory_rec_loss.append(r)
        theory_kl_loss.append(k)

    plt.plot(data['beta'], theory_total_loss, label='total')
    plt.plot(data['beta'], theory_rec_loss, label='rec')
    plt.plot(data['beta'], theory_kl_loss, label=r'$\beta$ KL')

    plt.legend()
    plt.xlabel(r"$\beta$")    
    plt.ylabel("loss")
    plt.savefig(f'output/{args.output_prefix}_loss.pdf')



def plot_beta_VAE_model(args):
    pass

if __name__ == "__main__":
    args = parser.parse_args()
    data = pd.read_csv(args.input_file)

    if args.linear:
        plot_linear_beta_VAE_model(args)
