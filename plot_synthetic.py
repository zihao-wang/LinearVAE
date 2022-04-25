import matplotlib.pyplot as plt

import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--input_file', type=str)
parser.add_argument('--output_prefix', type=str)
parser.add_argument('--m', default=5, type=int)
parser.add_argument('--M', default=5, type=int)
parser.add_argument('--guidelines', action='store_true')
parser.add_argument('--theory_loss', action='store_true')
parser.add_argument('--theory_sigma', action='store_true')

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
figsize = (3.5, 2.5)


# xi_list = [1, 2, 3, 4, 5]
# xi_list = [3.2911, 2.9455, 2.4364, 2.2479, 1.7337] # linear
# xi2_list = [xi ** 2 for xi in xi_list]

# print(xi_list)

# xi2_list = [11.8407, 10.2102,  5.8561,  4.9906,  2.4817] # regression relu
xi2_list = [5.11678773, 3.74132848, 3.25265424, 2.84157334, 2.56707496, 35.20561675215616] # mnist top 5



def theory_losses(beta):
    total_loss = 0
    rec_loss = 0
    kl_loss = 0
    sigma_list = []
    for i, xi2 in enumerate(xi2_list):
        if i >= args.m:
            rec_loss += xi2
        else:
            if beta < xi2:
                sigma = np.sqrt(beta / xi2)
            else:
                sigma = 1
            sigma_list.append(sigma)
            rec_loss += xi2 - max(0, np.sqrt(xi2) - np.sqrt(beta) * sigma) ** 2
            kl_loss += beta * (sigma ** 2 - 1 - 2 * np.log(sigma + 1e-10))

    total_loss = rec_loss + kl_loss
    return total_loss / 2


def plot_loss(args):
    data = pd.read_csv(args.input_file)
    plt.figure(figsize=figsize)

    plt.scatter(data['beta'], data['rec_loss'], marker='d',
                color=colors[1], alpha=0.8, label=r'$\ell_{rec}$')
    plt.scatter(data['beta'], data['kl_loss'], marker='D',
                color=colors[2], alpha=0.8, label=r'$\beta \ell_{KL}$')
    plt.scatter(data['beta'], data['total_loss'], marker='s',
                color=colors[0], alpha=0.8, label=r'$L_{\rm VAE}$')
    # if args.linear:
    # plt.scatter(data['beta'], data['enc_norm'], marker='>', color=colors[3], alpha=0.8, label=r'$\|W\|_F$')
    # plt.scatter(data['beta'], data['dec_norm'], marker='<', color=colors[4], alpha=0.8, label=r'$\|U\|_F$')


    if args.theory_loss:
        beta_list = data['beta'].tolist()
        theory_beta_list = np.linspace(0, max(beta_list), 100).tolist()
        theory_total_loss = []
        for beta in theory_beta_list:
            t = theory_losses(beta)
            theory_total_loss.append(t)
        plt.plot(theory_beta_list, theory_total_loss, alpha=.5, color='black')
    # plt.plot(data['beta'], theory_rec_loss, label='rec')
    # plt.plot(data['beta'], theory_kl_loss, label=r'$\beta$ KL')
    if args.guidelines:
        plt.vlines(x=xi2_list[:5], ymin=0, ymax=28,
                   color='gray', alpha=.5, ls='--')

    plt.legend()
    plt.xlabel(r"$\beta$")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(f'output/{args.output_prefix}_loss.pdf')


def plot_variance(args):
    data = pd.read_csv(args.input_file)

    m = args.m
    n = len(data)

    sigma_mean_mat = np.zeros((args.M, n))
    sigma_std_mat = np.zeros((args.M, n))

    for i in range(args.M):
        sigma_mean_mat[i] = data[f'sigma-{i}_mean']
        sigma_std_mat[i] = data[f'sigma-{i}_std']

    for i in range(n):
        sids = np.argsort(sigma_mean_mat[:, i])
        sigma_mean_mat[:, i] = sigma_mean_mat[sids, i]
        sigma_std_mat[:, i] = sigma_std_mat[sids, i]

    theory_beta_array = np.linspace(0, data['beta'], 100)

    plt.figure(figsize=figsize)
    markers = ["s", "^", "v", "<", ">"]
    for _i in range(m):
        i = 4-_i
        if args.theory_sigma:
            plt.plot(theory_beta_array,
                     np.minimum(1, np.sqrt(theory_beta_array / xi2_list[i])),
                     alpha=.5, color='black')
        plt.scatter(data['beta'], sigma_mean_mat[i],
                    marker=markers[i % 5], alpha=.8, color=colors[i % 5], label=rf'$\bar \sigma_{i+1}$')
    if args.guidelines:
        plt.vlines(x=xi2_list[: m], ymin=0, ymax=1, color='gray', alpha=.5, ls='--')
    plt.legend()
    # plt.ylim([0.6, 1.02])

    plt.xlabel(r"$\beta$")
    plt.ylabel(r"Encoding std")
    plt.tight_layout()
    plt.savefig(f'output/{args.output_prefix}_sigma.pdf')

    plt.figure(figsize=figsize)
    for i in range(m):
        plt.plot(data['beta'], sigma_std_mat[i],
                 marker='d', alpha=.8, color=colors[i % 5], label=rf'std $\sigma_{i+1}$')
    if args.guidelines:
        plt.vlines(x=xi2_list[: m], ymin=0, ymax=.01,
                   color='gray', alpha=.5, ls='--')
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
