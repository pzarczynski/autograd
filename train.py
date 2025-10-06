import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy import random

from src import *


class mlp(module):
    def __init__(self, dims: list[int] = [784, 10]) -> None:
        super().__init__(compact=True)
        assert len(dims) >= 2
        self.dims = dims
        self.layers = module_list()
        for in_feat, out_feat in zip(dims[:-1], dims[1:]):
            self.layers += dense(in_feat, out_feat)
    
    def forward(self, x: variable) -> variable:
        for layer in self.layers[:-1]: 
            x = relu()(layer(x))
        x = self.layers[-1](x)
        return x
    
    def __repr__(self):
        return f"mlp({', '.join(map(str, self.dims))})"
    
    
class encoder(module):
    def __init__(self, input_size: int, dims: list[int], latent_dim: int) -> None:
        super().__init__()
        self.mlp_block = mlp([input_size] + dims)
        self.mu_proj = dense(dims[-1], latent_dim)
        self.logvar_proj = dense(dims[-1], latent_dim)
        
    def forward(self, x: variable):
        x = relu()(self.mlp_block(x))
        mu = self.mu_proj(x).set_name("mu")
        logvar = self.logvar_proj(x).set_name("logvar")
        return mu, logvar
    
    
class decoder(module):
    def __init__(self, output_size: int, dims: list[int], latent_dim: int) -> None:
        super().__init__()
        self.mlp_block = mlp([latent_dim] + dims + [output_size])
        
    def forward(self, x: variable):
        return self.mlp_block(x)
    
    
class vae(module):
    def __init__(
        self, 
        input_size: int,
        encoder_dims: list[int], 
        decoder_dims: list[int], 
        latent_dim: int
    ) -> None:
        super().__init__()
        self.encoder = encoder(input_size=input_size, dims=encoder_dims, latent_dim=latent_dim)
        self.decoder = decoder(output_size=input_size, dims=decoder_dims, latent_dim=latent_dim)
        self.latent_dim = latent_dim
    
    @compact("kl_loss")
    def kl_divergence(mu: variable, logvar: variable) -> variable:
        return 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(axis=-1).mean()
    
    @compact("reparam")
    def reparameterize(mu: variable, logvar: variable) -> variable:
        return mu + (logvar * 0.5).exp() * random.randn(*mu.shape)
    
    def forward(self, x):
        x = variable(x)
        mu, logvar = self.encoder(x)
        kl_loss = self.kl_divergence(mu, logvar)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return out, kl_loss, mu
    
    def __repr__(self) -> str: return "vae()"


def train_one_epoch(
    model: vae, 
    dataset: tuple[Array, Array],  
    optim: adam, 
    batch_size: int = 256,
    beta: float = 1.0,
    graph: bool = False
) -> float:
    n_batches = dataset[0].shape[0] // batch_size
    run_recon_loss = run_kl_loss = 0
    
    for x, _ in batchify(*dataset, batch_size=batch_size):
        x_recon, kl_loss, _ = model(x)
        recon_loss = bce_with_logits_loss(x_recon, x)
        loss = recon_loss + kl_loss * beta
        
        if graph: save_graph(create_nx_graph(loss))
        
        loss.backward()
        optim.update()
        optim.zero_grad()
        run_recon_loss += float(recon_loss)
        run_kl_loss += float(kl_loss)
        
    run_recon_loss /= n_batches
    run_kl_loss /= n_batches
    return run_recon_loss, run_kl_loss


def val_one_epoch(
    model: vae,
    dataset: tuple[Array, Array],
    optim: adam,
    batch_size: int = 64,
    num_samples: int = 0,
) -> float:
    running_loss = 0
    n_batches = dataset[0].shape[0] // batch_size
    latents = list()
    
    for x, y in batchify(*dataset, batch_size=batch_size):
        x_recon, kl_loss, mu = model(x)
        latents.append((mu, y))
        
        recon_loss = bce_with_logits_loss(x_recon, x)
        running_loss += float(recon_loss + kl_loss)
        
    optim.zero_grad()
    
    out = [running_loss / n_batches]
    latents, labels = zip(*latents)
    out.append((np.concat(latents), np.concat(labels)))
    
    if num_samples > 0:
        z = random.randn(num_samples, model.latent_dim)
        out.append(sigmoid()(model.decoder(z)))  
    
    return tuple(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=10)
    parser.add_argument('--encoder-dims', type=list, default=[256])
    parser.add_argument('--decoder-dims', type=list, default=[256])
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--graph', type=bool, default=True)
    
    args = parser.parse_args()
    g = args.graph
    
    train_ds, val_ds = load_mnist()
    
    model = vae(
        input_size=train_ds[0].shape[-1], encoder_dims=args.encoder_dims, 
        decoder_dims=args.decoder_dims, latent_dim=args.latent_dim
    )
    tx = adam(model.params(), lr=args.lr)
    schedule = cosine_decay_schedule(args.lr, 1e-4, args.epochs)
    
    for epoch in range(args.epochs):
        train_ds = shuffle(*train_ds)
        tx.lr = schedule(epoch)
        
        r_loss, kl_loss = train_one_epoch(model, train_ds, tx, args.bs, beta=args.beta, graph=g)
        g = False
        
        val_loss, (latents, labels), samples = val_one_epoch(model, val_ds, tx, args.bs, num_samples=16)
        
        print("Epoch: {};\trecon_loss: {:.2f};\tkl_loss: {:.2f};\tval_loss: {:.2f};\tlr: {:.3e}"
              .format(epoch + 1, r_loss, kl_loss, val_loss, tx.lr))
        
        save_samples(samples, ncols=4)
        
        if (epoch + 1) % 5 == 0:
            visualize_space(latents, labels)
        
    model.save("model.pkl")
