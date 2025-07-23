import os
import numpy as np
from src import *

NX_GRAPH = True


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
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.mlp_block = mlp(dims[:-1])
        self.mu_proj = dense(dims[-2], dims[-1])
        self.logvar_proj = dense(dims[-2], dims[-1])
        
    def forward(self, x: variable):
        x = relu()(self.mlp_block(x))
        mu = self.mu_proj(x).set_name("mu")
        logvar = self.logvar_proj(x).set_name("logvar")
        return mu, logvar
    
    
class decoder(module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.mlp_block = mlp(dims)
        
    def forward(self, x: variable):
        return self.mlp_block(x)
    
    
class vae(module):
    def __init__(self, encoder_dims: list[int], decoder_dims: list[int]):
        super().__init__()
        self.encoder = encoder(encoder_dims)
        self.decoder = decoder(decoder_dims)
    
    @compact("kl_divergence()")
    def kl_divergence(mu: variable, logvar: variable) -> variable:
        loss = (logvar.exp() + mu.pow(2) - 1.0 - logvar) * 0.5
        return loss.mean()
    
    @compact("reparametrization()")
    def reparameterize(mu: variable, logvar: variable) -> variable:
        return mu + (logvar * 0.5).exp() * np.random.randn(*mu.shape)
    
    def forward(self, x, return_latents=False):
        mu, logvar = self.encoder(x)
        kl_loss = self.kl_divergence(mu, logvar)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        if return_latents: return out, kl_loss, mu
        return out, kl_loss
    
    def __repr__(self) -> str: return "vae()"


def train_one_epoch(
    model: vae, 
    dataset: tuple[Array, ...],  
    optim: adam, 
    batch_size: int = 64,
    beta: float = 1.0,
) -> float:
    global NX_GRAPH
    n_batches = dataset[0].shape[0] // batch_size
    run_recon_loss = run_kl_loss = 0
    
    for x, *_ in batchify(*dataset, batch_size=batch_size):
        x_recon, kl_loss = model(variable(x))
        recon_loss = bce_with_logits_loss(x_recon, x)
        loss = recon_loss + kl_loss * beta
        
        if NX_GRAPH:
            G, _ = create_nx_graph(loss)
            visualize_dag(G)
            NX_GRAPH = False
        
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
    dataset: tuple[Array, ...],
    optim: adam,
    batch_size: int = 64,
    num_samples: int = 4,
    return_latents: bool = False,
) -> float:
    running_loss = 0
    n_batches = dataset[0].shape[0] // batch_size
    latents = list()
    
    for x, *y in batchify(*dataset, batch_size=batch_size):
        if return_latents:
            x_recon, kl_loss, new_latents = model(variable(x), return_latents=True)
            latents.append((new_latents, y[0][..., 0]))
        else: x_recon, kl_loss = model(variable(x))
        recon_loss = bce_with_logits_loss(x_recon, x)
        loss = recon_loss + kl_loss
        optim.zero_grad()
        running_loss += float(loss)
    
    out = [running_loss / n_batches]
    if num_samples > 0:
        z = np.random.randn(num_samples, 32)
        samples = model.decoder(z)
        out.append(samples)  
    if return_latents: 
        latents = tuple(map(np.concat, zip(*latents)))
        out.append(latents)
    return tuple(out)
        

def train_loop(num_epochs: int):
    os.makedirs("img", exist_ok=True)
    
    (x_train, _), (x_test, y_test) = load_mnist()
    model = vae(
        encoder_dims=[784, 256, 32], 
        decoder_dims=[32, 256, 784],
    )
    optim = adam(model.params(), lr=1e-3)
    beta_scheduler = cosine_annealing_warm_restarts(
        min_eta=0.01,
        max_eta=0.6, 
        T_0=5,
        T_mul=2,
        invert=True
    )
    
    for epoch in range(num_epochs):
        x_train, = shuffle(x_train)
        
        beta = beta_scheduler(epoch)
        print(f"beta: {beta:.3f}")
        recon_loss, kl_loss = train_one_epoch(model, (x_train,), optim, beta=beta)
        
        val_loss, samples = val_one_epoch(model, (x_test,), optim, num_samples=16)
        print(format_info(epoch, {'kl': kl_loss, 'recon': recon_loss}, val_loss))
        
        visualize_samples(4, samples).savefig(f"img/epoch_{epoch+1}_samples")
        
    model.save("model.pkl")
    
    *_, (latents, labels) = val_one_epoch(model, (x_test, y_test), optim, return_latents=True)
    visualize_space(latents, labels).savefig(f"img/latent_space")
    

if __name__ == '__main__':
    train_loop(5)