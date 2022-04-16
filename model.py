from re import M
import torch
from torch import nn, relu
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal


class LinearVariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(LinearVariationalEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nn_mean = nn.Linear(input_dim, latent_dim, bias=False)
        self.nn_logvar =  nn.Linear(input_dim, latent_dim, bias=True)
    
    def forward(self, x):
        batch_size = x.size(0)
        eps = torch.randn(batch_size, self.latent_dim)
        mu = self.nn_mean(x)
        logvar = self.nn_logvar(x)
        sigma = logvar.div(2).exp()
        return {'z': mu + sigma * eps,
                'mu': mu,
                'sigma': sigma}

class LinearVariationalDecoder(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super(LinearVariationalDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.l = nn.Linear(latent_dim, target_dim, bias=False)

    def forward(self, z):
        y = self.l(z)
        return y

class LinearBetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, target_dim, eta_dec_sq, eta_prior_sq, beta, **kwargs):
        super(LinearBetaVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.eta_dec_sq = eta_dec_sq
        self.eta_prior_sq = eta_prior_sq
        self.beta = beta

        self.encoder = LinearVariationalEncoder(input_dim, latent_dim)
        self.decoder = LinearVariationalDecoder(latent_dim, target_dim)

    def forward(self, x, y):
        encoded = self.encoder(x)
        y_pred = self.decoder(encoded['z'])
        rec_loss = torch.square(y - y_pred).sum(-1).mean(0) / self.eta_dec_sq / 2

        kl_loss = .5 * (
            - torch.log(encoded['sigma']**2 / self.eta_prior_sq).sum(-1) 
            - self.latent_dim
            + torch.norm(encoded['mu'], p=2, dim=-1) ** 2 / self.eta_prior_sq
            + (encoded['sigma'] ** 2 / self.eta_prior_sq).sum(-1)
            ).mean(0)
        loss = rec_loss + self.beta * kl_loss
        forward_dict = {
            'z': encoded['z'],
            'mu': encoded['mu'],
            'sigma': encoded['sigma'],
            'y_pred': y_pred,
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss * self.beta,
            'enc_norm': self.encoder.nn_mean.weight.norm(),
            'dec_norm': self.decoder.l.weight.norm()
        }

        return forward_dict


class ReLUVariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(ReLUVariationalEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.nn_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
            )
        self.nn_logvar =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        eps = torch.randn(batch_size, self.latent_dim)
        mu = self.nn_mean(x)
        logvar = self.nn_logvar(x)
        sigma = logvar.div(2).exp()
        return {'z': mu + sigma * eps,
                'mu': mu,
                'sigma': sigma}

class ReLUVariationalDecoder(nn.Module):
    def __init__(self, latent_dim, target_dim, hidden_dim):
        super(ReLUVariationalDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.l = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, z):
        y = self.l(z)
        return y

class ReLUBetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, target_dim, hidden_dim, eta_dec_sq, eta_prior_sq, beta, **kwargs):
        super(ReLUBetaVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.eta_dec_sq = eta_dec_sq
        self.eta_prior_sq = eta_prior_sq
        self.beta = beta

        self.encoder = ReLUVariationalEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = ReLUVariationalDecoder(latent_dim, target_dim, hidden_dim)

    def forward(self, x, y):
        encoded = self.encoder(x)
        y_pred = self.decoder(encoded['z'])
        rec_loss = torch.square(y - y_pred).sum(-1).mean(0) / self.eta_dec_sq / 2

        kl_loss = .5 * (
            - torch.log(encoded['sigma']**2 / self.eta_prior_sq).sum(-1) 
            - self.latent_dim
            + torch.norm(encoded['mu'], p=2, dim=-1) ** 2 / self.eta_prior_sq
            + (encoded['sigma'] ** 2 / self.eta_prior_sq).sum(-1)
            ).mean(0)
        loss = rec_loss + self.beta * kl_loss
        forward_dict = {
            'z': encoded['z'],
            'mu': encoded['mu'],
            'sigma': encoded['sigma'],
            'y_pred': y_pred,
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss * self.beta,
            # 'enc_norm': self.encoder.nn_mean.weight.norm(),
            # 'dec_norm': self.decoder.l.weight.norm()
        }

        return forward_dict



class TanhVariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(TanhVariationalEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.nn_mean = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
            )
        self.nn_logvar =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        eps = torch.randn(batch_size, self.latent_dim)
        mu = self.nn_mean(x)
        logvar = self.nn_logvar(x)
        sigma = logvar.div(2).exp()
        return {'z': mu + sigma * eps,
                'mu': mu,
                'sigma': sigma}

class TanhVariationalDecoder(nn.Module):
    def __init__(self, latent_dim, target_dim, hidden_dim):
        super(TanhVariationalDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.l = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, z):
        y = self.l(z)
        return y

class TanhBetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, target_dim, hidden_dim, eta_dec_sq, eta_prior_sq, beta, **kwargs):
        super(TanhBetaVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.eta_dec_sq = eta_dec_sq
        self.eta_prior_sq = eta_prior_sq
        self.beta = beta

        self.encoder = TanhVariationalEncoder(input_dim, latent_dim, hidden_dim)
        self.decoder = TanhVariationalDecoder(latent_dim, target_dim, hidden_dim)

    def forward(self, x, y):
        encoded = self.encoder(x)
        y_pred = self.decoder(encoded['z'])
        rec_loss = torch.square(y - y_pred).sum(-1).mean(0) / self.eta_dec_sq / 2

        kl_loss = .5 * (
            - torch.log(encoded['sigma']**2 / self.eta_prior_sq).sum(-1) 
            - self.latent_dim
            + torch.norm(encoded['mu'], p=2, dim=-1) ** 2 / self.eta_prior_sq
            + (encoded['sigma'] ** 2 / self.eta_prior_sq).sum(-1)
            ).mean(0)
        loss = rec_loss + self.beta * kl_loss
        forward_dict = {
            'z': encoded['z'],
            'mu': encoded['mu'],
            'sigma': encoded['sigma'],
            'y_pred': y_pred,
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss * self.beta,
            # 'enc_norm': self.encoder.nn_mean.weight.norm(),
            # 'dec_norm': self.decoder.l.weight.norm()
        }

        return forward_dict