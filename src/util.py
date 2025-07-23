import numpy as np
import os

from numpy.typing import NDArray as Array

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from .ops import variable

from typing import Iterable
from sklearn.manifold import TSNE

import seaborn as sns
sns.set_theme()


def load_mnist(
    root_dir: str = './tmp',
    test_size: float = 0.2
) -> tuple[tuple[Array, Array],  tuple[Array, Array]]:
    os.makedirs(root_dir, exist_ok=True)
    path = os.path.join(root_dir, 'mnist.npz')
    try:
        archive = np.load(path)
        x, y = archive['x'], archive['y']
    except FileNotFoundError:
        mnist = fetch_openml('mnist_784', as_frame=False, parser='liac-arff')
        x = mnist.data.astype(np.float32) / 255
        y = mnist.target.astype(np.int64)
        np.savez(path, x=x, y=y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


def shuffle(*xs: Array) -> tuple[Array, ...]:
    idx = np.random.permutation(xs[0].shape[0])
    return tuple(x[idx] for x in xs)


def batchify(*xs: Array, batch_size: int = 64) -> Iterable[tuple[Array, ...]]:
    n_batches = xs[0].shape[0] // batch_size
    batched = [x[:n_batches*batch_size].reshape(n_batches, batch_size, -1) for x in xs]
    yield from zip(*batched)


def encode_one_hot(labels: Array, num_classes: int):
    B = labels.shape[0]
    encoded = np.zeros((B,  num_classes))
    encoded[np.arange(B), labels] = 1
    return encoded


def create_nx_graph(node: variable, visited: set = None) -> tuple[nx.DiGraph, set]:
    graph = nx.DiGraph()
        
    if visited is None:
        visited = set()
    visited.add(node)
    
    if node.grad_op is None:
        graph.add_node(id(node)-1) #, label="input")
    else:
        if node._module_info is not None:
            op = node._module_info['ref']
            parents = node._module_info['parents']
        else:
            op = node.grad_op
            parents = node.grad_op.parents
            
        graph.add_node(id(op), label=repr(op))
            
        for p in parents:
            if p not in visited:
                subgraph, visited = create_nx_graph(p, visited)
                graph.add_nodes_from(subgraph.nodes(data=True))
                graph.add_edges_from(subgraph.edges(data=True))
                
            prev_op = id(p.grad_op)
            if p.grad_op is None: prev_op = id(p) - 1
            elif p._module_info is not None: prev_op = id(p._module_info['ref'])
            
            label = p.name or ''
            if p.ndim != 0: label += str(p.shape)      
            graph.add_edge(prev_op, id(op), label=label)
    return graph, visited


def visualize_dag(graph: nx.DiGraph):
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
    net.show("graph.html", notebook=False)
    
    
class cosine_annealing_warm_restarts:
    def __init__(
        self, 
        min_eta: float = 0, 
        max_eta: float = 1, 
        T_0: int = 10, 
        T_mul: int = 2, 
        invert: bool = False
    ):
        self.eta = (min_eta, max_eta)
        self.T_0 = T_0
        self.T_mul = T_mul
        
        self.T = T_0
        self.T_start = 0
        self.invert = invert
        
    def __call__(self, t: int) -> float:
        min_eta, max_eta = self.eta
        t -= self.T_start
        
        if t >= self.T:
            self.T_start += self.T
            self.T *= self.T_mul
            t -= self.T_start
        
        mul_term = (1 + np.cos(np.pi * t / self.T)) / 2
        if self.invert: mul_term = 1 - mul_term
        
        eta = min_eta + (max_eta - min_eta) * mul_term
        return eta
    
    
def format_info(epoch: int, train_losses: float | dict[str, float], val_losses: float | dict[str, float]) -> str:
    info = f"epoch {epoch+1}:"
    if isinstance(train_losses, float): info += f" train_loss\t\t{train_losses:.3f};"
    else: info += ''.join(f" train_loss_{k}\t\t{v:.3f};" for k, v in train_losses.items())
    if isinstance(val_losses, float): info += f" val_loss\t\t{val_losses:.3f};"
    else: info += ''.join(f" val_loss_{k}\t\t{v:.3f};" for k, v in val_losses.items())
    return info


def visualize_space(latents: Array, labels: Array):
    print("running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    latent_2d = tsne.fit_transform(latents)
    print("done...")
    fig = plt.figure(figsize=(10, 8))
    ax = sns.kdeplot(
        x=latent_2d[:, 0],
        y=latent_2d[:, 1],
        hue=labels,
        fill=True,
        alpha=0.5,
        levels=10,
        palette='tab10',
        legend=True
    )
    ax.set_title('visualization of latent space')
    return fig


def visualize_samples(ncols: int, samples: list[Array]):
    assert len(samples) % ncols == 0
    nrows = len(samples) // ncols
    fig, axarr = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    for ax, sample in zip(axarr.ravel(), samples):
        ax.imshow(sample.reshape(28, 28))
        ax.axis('off')
    fig.tight_layout()
    return fig