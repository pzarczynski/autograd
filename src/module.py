import pickle
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from .ops import variable


def kaiming_normal(shape, fan_in):
    """Kaiming normal distribution (stddev = 2 / sqrt(`fan_in`))."""
    stddev = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * stddev


class module(ABC):
    def __init__(self, compact: bool = False) -> None:
        self.compact = compact
    
    def __call__(self, *parents, **kwargs) -> variable | tuple[variable, ...]:
        out = self.forward(*parents, **kwargs)
        if self.compact:
            module_info = dict(ref=self, parents=parents)
            if isinstance(out, variable):
                out._module_info = module_info
            else:
                for v in out: v._module_info = module_info
        return out    
    
    def save(self, fname) -> None:
        with open(fname, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, fname) -> None:
        with open(fname, "rb") as f:
            return pickle.load(f)
    
    def params(self) -> Iterable[variable]:
        for v in self.__dict__.values():
            if isinstance(v, variable):
                yield v
            elif isinstance(v, module):
                yield from v.params()
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> variable | Iterable[variable]: ...


class module_list(module):
    def __init__(self, *modules: module) -> None:
        super().__init__()
        self.len = len(modules)
        for i, m in enumerate(modules):
            self.__setattr__(str(i), m)
    
    def __iadd__(self, m: module) -> 'module_list':
        self.__setattr__(str(self.len), m)
        self.len += 1
        return self
    
    def __len__(self) -> int:
        return self.len
    
    def __iter__(self) -> Iterable[module]:
        for i in range(len(self)):
            yield getattr(self, str(i))
            
    def __getitem__(self, idx) -> module:
        return list(iter(self))[idx]
    
    def forward(self): ...


class dense(module):     
    def __init__(self, fan_in: int, fan_out: int):
        super().__init__(compact=True)
        self.weights = variable(kaiming_normal((fan_in, fan_out), fan_in), name="weights")
        self.bias = variable(np.zeros(fan_out), name="bias")
    
    def forward(self, x: variable) -> Iterable[variable]:
        return x @ self.weights + self.bias
        
    def __repr__(self) -> str:
        return "dense({}, {})".format(*self.weights.shape)


class optimizer:
    def __init__(self, params: Iterable[variable], lr: float):
        self.lr = lr
        self.params = tuple(params)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.fill(0)
                param._visited = 0
                param.out_degrees = 0
                
    def update(self): ...
    

class sgd(optimizer):        
    def update(self):
        for param in self.params:
            if param.grad is not None:
                grad = param.grad
                param = param.view(np.ndarray)
                param -= self.lr * grad
                

class adam(optimizer):
    def __init__(self, params: Iterable[variable], lr: float = 1e-3, betas=[0.9, 0.999], eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
    
        self.m = [np.zeros(p.shape, p.dtype) for p in self.params]
        self.v = [np.zeros(p.shape, p.dtype) for p in self.params]

    def update(self):
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is not None:
                g_t = param.grad.view(np.ndarray)
                param = param.view(np.ndarray)

                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g_t
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (g_t ** 2)

                m_hat_t = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat_t = self.v[i] / (1 - self.betas[1] ** self.t)

                param -= self.lr * m_hat_t / np.sqrt(v_hat_t + self.eps)


def compact(name: str = None):
    def wrapper(fn):
        class compact_module(module):
            def forward(sefl, *args, **kwargs): return fn(*args, **kwargs)
            def __repr__(self): return (name or fn.__repr__)
        return compact_module(compact=True)
    return wrapper