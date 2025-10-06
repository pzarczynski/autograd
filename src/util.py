from typing import Iterable

import math
import h5py
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from umap import UMAP
from numpy.typing import NDArray as Array
from pyvis.network import Network
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from .ops import variable

matplotlib.use('Agg')
sns.set_theme()


def load_mnist(file: str = 'mnist.hdf', test_size: float = 1 / 6
) -> tuple[tuple[Array, Array],  tuple[Array, Array]]:
    with h5py.File(file, 'a') as f:
        try:
            x, y = f['x'][:], f['y'][:]
        except:
            mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
            x = np.where(mnist.data > 128, 1, 0).astype(np.float32)
            # x = mnist.data.astype(np.float32) / 255
            y = mnist.target.astype(np.int64)
            f.create_dataset('x', data=x)
            f.create_dataset('y', data=y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


def shuffle(x: Array, y: Array) -> tuple[Array, ...]:
    idx = np.random.permutation(x.shape[0])
    return x[idx], y[idx]


def batchify(x: Array, y: Array, batch_size: int) -> Iterable[tuple[Array, Array]]:
    n_batches = x.shape[0] // batch_size
    x = x[:n_batches*batch_size].reshape(n_batches, batch_size, *x.shape[1:])
    y = y[:n_batches*batch_size].reshape(n_batches, batch_size, *y.shape[1:])
    yield from zip(x, y)


def encode_one_hot(labels: Array, num_classes: int):
    B = labels.shape[0]
    encoded = np.zeros((B,  num_classes))
    encoded[np.arange(B), labels] = 1
    return encoded


def cosine_decay_schedule(peak_value: int, end_value: int, T: int):
    def fn(t: int):
        return end_value + (peak_value - end_value) * (1 + math.cos(math.pi * t / (T - 1))) / 2
    return fn


def save_samples(samples: list[Array], ncols: int) -> None:
    nrows = len(samples) // ncols
    fig, axarr = plt.subplots(nrows, ncols)
    
    for ax, sample in zip(axarr.ravel(), samples):
        ax.imshow(sample.reshape(28, 28), cmap='gray')
        ax.axis('off')
        
    fig.tight_layout()
    fig.savefig('samples')
    plt.close(fig)


def visualize_space(latents: Array, labels: Array, num_samples: int = 2000) -> None:
    idx = np.random.randint(0, latents.shape[0], size=num_samples)
    sample_latents, sample_labels = latents[idx], labels[idx]
    
    print("Running UMAP...")
    latent_2d = UMAP().fit_transform(sample_latents)
    
    # print("Running t-SNE...")
    # latent_2d = TSNE(perplexity=50, n_jobs=-1).fit_transform(sample_latents)
    print("Done...")

    ax = sns.kdeplot(
        x=latent_2d[:, 0],
        y=latent_2d[:, 1],
        hue=sample_labels,
        fill=True,
        alpha=0.8,
        levels=10,
        palette='tab10',
        legend=True
    )
    ax.set_title('Visualization of the latent space')
    plt.tight_layout()
    plt.savefig('latent_space')
    plt.close()


def create_nx_graph(node: variable) -> nx.DiGraph:
    def _create_nx_graph(n: variable, v: set):
        g = nx.DiGraph()
        v.add(n)
        
        if n.grad_op is None:
            g.add_node(id(n)-1) #, label="input")
        else:
            if n._module_info is not None:
                op = n._module_info['ref']
                parents = n._module_info['parents']
            else:
                op = n.grad_op
                parents = n.grad_op.parents
                
            g.add_node(id(op), label=repr(op))
                
            for p in parents:
                if p not in v:
                    subgraph, v = _create_nx_graph(p, v)
                    g.add_nodes_from(subgraph.nodes(data=True))
                    g.add_edges_from(subgraph.edges(data=True))
                    
                prev_op = id(p.grad_op)
                if p.grad_op is None: prev_op = id(p) - 1
                elif p._module_info is not None: prev_op = id(p._module_info['ref'])
                
                label = p.name or ''
                if p.ndim != 0: label += str(p.shape)      
                g.add_edge(prev_op, id(op), label=label)         
        return g, v
    
    return _create_nx_graph(node, set())[0]


def save_graph(graph: nx.DiGraph):
    net = Network(directed=True, height="800px")
    net.from_nx(graph)
    net.set_options("""
    {
        "layout": {
            "hierarchical": {
                "enabled": true,
                "levelSeparation": 150,
                "nodeSpacing": 100,
                "direction": "LR",
                "sortMethod": "directed"
            }
        },
        "physics": {
            "enabled": false
        },
        "edges": {
            "smooth": {
                "enabled": true,
                "type": "cubicBezier",
                "roundness": 0.5
            }
        },
        "nodes": {
            "font": {
                "size": 12,
                "color": "black"
            }
        }
    }
    """)
    net.save_graph("graph.html")
