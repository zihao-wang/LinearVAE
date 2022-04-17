import matplotlib.pyplot as plt

import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_prefix', type=str)
parser.add_argument('--linear', action='store_true')
parser.add_argument('--m', default=5, type=int)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
figsize = (3.5, 2.5)

def theory_losses(beta, xi_list):
    total_loss = 0
    rec_loss = 0
    kl_loss = 0
    sigma_list = []
    for i, xi in enumerate(xi_list):
        if len(xi_list) - i > args.m:
            rec_loss += xi ** 2
        else:
            if beta < xi ** 2:
                sigma = np.sqrt(beta) / xi
            else:
                sigma = 1
            sigma_list.append(sigma)
            rec_loss += xi ** 2 - max(0, xi - np.sqrt(beta) * sigma) ** 2
            kl_loss  += beta * (sigma ** 2 - 1 - 2 * np.log(sigma + 1e-10))
    
    total_loss = rec_loss + kl_loss
    return total_loss / 2


def plot_loss(args):
    data = pd.read_csv(args.input_file)
    plt.figure(figsize=figsize)

    plt.scatter(data['beta'], data['rec_loss'], marker='d', color=colors[1], alpha=0.8, label='rec')
    plt.scatter(data['beta'], data['kl_loss'], marker='D', color=colors[2], alpha=0.8, label=r'$\beta$ KL')
    plt.scatter(data['beta'], data['total_loss'], marker='s', color=colors[0], alpha=0.8, label='total')
    if args.linear:
        plt.scatter(data['beta'], data['enc_norm'], marker='>', color=colors[3], alpha=0.8, label=r'$\|W\|_F$')
        plt.scatter(data['beta'], data['dec_norm'], marker='<', color=colors[4], alpha=0.8, label=r'$\|U\|_F$')

    beta_list = data['beta'].tolist()
    xi_list = [1, 2, 3, 4, 5]

    theory_total_loss = []
    for beta in beta_list:
        t = theory_losses(beta, xi_list)
        theory_total_loss.append(t)

    plt.plot(data['beta'], theory_total_loss, color=colors[0])
    # plt.plot(data['beta'], theory_rec_loss, label='rec')
    # plt.plot(data['beta'], theory_kl_loss, label=r'$\beta$ KL')
    plt.vlines(x=[1, 4, 9, 16, 25], ymin=0, ymax=30, color='gray', alpha=.5, ls='--')
    plt.legend()
    plt.xlabel(r"$\beta$")    
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f'output/{args.output_prefix}_loss.pdf')

def plot_variance(args):
    data = pd.read_csv(args.input_file)

    m = args.m
    n = len(data)

    sigma_mean_mat = np.zeros((m, n))
    sigma_std_mat  = np.zeros((m, n))

    for i in range(m):
        sigma_mean_mat[i] = data[f'sigma-{i}_mean']
        sigma_std_mat[i] = data[f'sigma-{i}_std']
    
    for i in range(n):
        sids = np.argsort(sigma_mean_mat[:, i], )[::-1]
        sigma_mean_mat[:, i] = sigma_mean_mat[sids, i]
        sigma_std_mat[:, i]  = sigma_std_mat[sids, i]

    plt.figure(figsize=figsize)
    markers = ["s", "^", "v", "<", ">"]
    for i in range(m):
        if args.linear:
            plt.plot(data['beta'], np.minimum(1, np.sqrt(data['beta']) / (i+1) ), 
                    color=colors[i])
        plt.scatter(data['beta'], sigma_mean_mat[i], 
                    marker=markers[i], alpha=.8, color=colors[i], label=rf'mean $\sigma_{i+1}$')
    plt.vlines(x=[1, 4, 9, 16, 25], ymin=0, ymax=1, color='gray', alpha=.5, ls='--')
    plt.legend()
    plt.xlabel(r"$\beta$")    
    plt.ylabel(r"mean $\sigma$")
    plt.tight_layout()
    plt.savefig(f'output/{args.output_prefix}_sigma.pdf')


    plt.figure(figsize=figsize)
    for i in range(m):
        plt.plot(data['beta'], sigma_std_mat[i],
                    marker='d', alpha=.8, color=colors[i], label=rf'std $\sigma_{i+1}$')
    plt.vlines(x=[1, 4, 9, 16, 25], ymin=0, ymax=0.025, alpha=0.5, ls='--')
    plt.xlabel(r"$\beta$")    
    plt.ylabel(r"$\sigma$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'output/{args.output_prefix}_sigma_std.pdf')


if __name__ == "__main__":
    args = parser.parse_args()
    data = pd.read_csv(args.input_file)

    plot_loss(args)
    plot_variance(args)
