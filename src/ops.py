from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray as Array


def reduce(array: Array, shape: tuple[int, ...]) -> Array:
    if array.shape == shape:
        return array
    
    ndim_diff = array.ndim - len(shape)
    shape_padded = (1,) * ndim_diff + shape
    
    axes = tuple(
        i for i, (b_dim, t_dim) in enumerate(zip(array.shape, shape_padded))
        if t_dim == 1 and b_dim > 1
    )
    reduced = np.add.reduce(array, axis=axes, keepdims=True)
    reduced = np.reshape(reduced, shape)
    return reduced    
    
    
class variable(np.ndarray):
    name: str
    grad: Optional[Array] 
    grad_op: Optional['operation']
    out_degrees: int
    _visited: int
    _module_info: Optional[dict]
    
    def __new__(cls, obj: Array, name: str = None) -> 'variable':
        if isinstance(obj, variable):
            return obj
        obj = np.asarray(obj).view(cls)
        obj.grad = None
        obj.grad_op = None
        obj.name = name
        obj.out_degrees = 0
        obj._visited = 0
        obj._module_info = None
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.grad = getattr(obj, 'grad', None)
        self.grad_op = getattr(obj, 'grad_op', None)
        self.name = getattr(obj, 'name', None)
        self.out_degrees = getattr(obj, 'out_degrees', 0)
        self._visited = getattr(obj, '_visited', 0)
        self._module_info = getattr(obj, '_module_info', 0)
    
    def backward(self) -> None:
        """Performs a backward pass using accumulated `grad`."""   
        if self.grad_op is not None:
            if self.grad is None: 
                self.grad = np.ones(self.shape, dtype=self.dtype)
            self.grad_op._backward(self.grad)
        
    def add_grad(self, grad: Array):
        if self.grad is None: 
            self.grad = np.zeros(self.shape, dtype=self.dtype)
        self.grad += grad
        self._visited += 1
        
    def set_name(self, name: str) -> 'variable':
        self.name = name
        return self
        
    def detach(self):
        return variable(np.asarray(self))
    
    def pow(self, power: Array):
        return pow()(self, power=power)
    
    def exp(self):
        return exp()(self)
    
    def sum(self, axis: int = None):
        return sum_along_axis()(self, axis=axis)
    
    def mean(self, axis: int = None):
        return mean()(self, axis=axis)
        
    def __add__(self, other):
        return add()(self, variable(other)) 
    
    def __iadd__(self, other):
        return add()(self, variable(other))
    
    def __neg__(self):
        return neg()(self)
    
    def __sub__(self, other):
        return sub()(self, variable(other))
    
    def __isub__(self, other):
        return sub()(self, variable(other))
    
    def __mul__(self, other):
        return mul()(self, variable(other))
    
    def __imul__(self, other):
        return mul()(self, variable(other))
    
    def __matmul__(self, other):
        return matmul()(self, variable(other))
    
    def __repr__(self) -> str:
        return f'variable(shape={self.shape}, grad_op={self.grad_op})'    
        
    def __str__(self) -> str:
        r = 'variable('
        r += self._indented_array_str(len(r))
        return r + ')'
    
    def _indented_array_str(self, indent: int) -> str:
        l = np.array_str(self).splitlines()
        l[1:] = [indent*' ' + x for x in l[1:]]
        return '\n'.join(l)
    
    def __hash__(self) -> int:
        return id(self)
    
    
class operation(ABC):
    parents: tuple[variable, ...]
    
    def __call__(self, *parents: variable, **kwargs) -> variable:
        self.parents = parents
        arrays = map(np.asarray, parents)
        out = variable(self.forward(*arrays, **kwargs))
        out.grad_op = self
        for p in parents:
            p.out_degrees += 1
        return out
        
    def _backward(self, grad: Array) -> None:
        xs = tuple(map(np.asarray, self.parents))
        grads = list(self.backward(xs, grad))
        for p, g in zip(self.parents, grads):
            p.add_grad(g)
            if p._visited == p.out_degrees:
                p.backward()  
             
    @abstractmethod
    def forward(self, *inputs: Array) -> Array: ...
        
    @abstractmethod
    def backward(self, xs: tuple[Array, ...], grad: Array) -> Iterable[Array]: ...
        

class matmul(operation):
    def forward(self, x: Array, y: Array) -> Array:
        return x @ y
    
    def backward(self, xs: tuple[Array, ...], grad: Array):
        x, y = xs
        yield from (grad @ y.T, x.T @ grad)
        
    def __repr__(self) -> str: return "matmul()"
        

class neg(operation):
    def forward(self, x: Array):
        return -x
    
    def backward(self, xs: tuple[Array, ...], grad):
        assert len(xs) == 1
        yield -grad
        
    def __repr__(self) -> str: return "neg()"


class add(operation):    
    def forward(self, *inputs: Array) -> Array:
        broadcasted = np.broadcast_arrays(*inputs)
        return np.sum(broadcasted, axis=0)
    
    def backward(self, xs: tuple[Array, ...], grad: Array) -> Iterable[Array]:
        yield from (reduce(grad, x.shape) for x in xs)
            
    def __repr__(self) -> str: return "add()"
    
    
class sub(operation):
    def forward(self, x, y) -> Array:
        return x - y
    
    def backward(self, xs, grad):
        yield reduce(grad, xs[0].shape)
        yield reduce(-grad, xs[1].shape)
    
    def __repr__(self) -> str: return "sub()"
            
            
class mul(operation):
    def forward(self, *inputs: Array):
        broadcasted = np.broadcast_arrays(*inputs)
        return np.prod(broadcasted, axis=0)
    
    def backward(self, xs: tuple[Array, ...], grad: Array) -> Iterable[Array]:    
        broadcasted = np.broadcast_arrays(*xs)
        for i, p in enumerate(self.parents):
            other = broadcasted[:i] + broadcasted[i+1:]
            prod = grad * np.prod(other, axis=0)
            yield reduce(prod, p.shape)
            
    def __repr__(self) -> str: return "mul()" 
    
    
class pow(operation):
    def forward(self, x: Array, *, power: Array):
        self.power = power
        return np.pow(x, power)
    
    def backward(self, xs: tuple[Array, ...], grad):
        assert len(xs) == 1
        yield grad * np.pow(xs[0], self.power - 1) * self.power
        
    def __repr__(self) -> str: return "pow()"
    
    
class sum_along_axis(operation):
    def forward(self, x: Array, *, axis: int):
        self.axis = axis
        return np.sum(x, axis=axis)
    
    def backward(self, xs: tuple[Array, ...], grad):
        assert len(xs) == 1
        if self.axis is None:
            yield np.full(xs[0].shape, grad)
        else:
            grad_expanded = np.expand_dims(grad, axis=self.axis)
            yield np.broadcast_to(grad_expanded, xs[0].shape)
            
    def __repr__(self) -> str: return "sum()"
        
        
class mean(operation):
    def forward(self, x: Array, *, axis: int):
        self.axis = axis
        return np.mean(x, axis=axis)

    def backward(self, xs: tuple[Array, ...], grad):
        assert len(xs) == 1
        if self.axis is None:
            yield np.full(xs[0].shape, grad / xs[0].size)
        else:
            grad_expanded = np.expand_dims(grad, axis=self.axis)
            yield np.broadcast_to(grad_expanded / xs[0].shape[self.axis], xs[0].shape)
    
    def __repr__(self) -> str: return "mean()"
        

class exp(operation):
    def forward(self, x: Array):
        self.exp = np.exp(x)
        return self.exp
    
    def backward(self, xs: tuple[Array, ...], grad):
        assert len(xs) == 1
        yield grad * self.exp
        
    def __repr__(self) -> str: return "exp()"


class binary_cross_entropy_with_logits(operation):
    def forward(self, logits: Array, targets: Array):
        log_sigmoid_logits = -np.logaddexp(0, -logits)
        log_one_minus_sigmoid_logits = -np.logaddexp(0, logits)
        loss_element_wise = -(targets * log_sigmoid_logits + 
                              (1 - targets) * log_one_minus_sigmoid_logits)
        loss = np.mean(np.sum(loss_element_wise, axis=-1))
        return loss
    
    def backward(self, xs, grad):
        sigmoid_logits = 1 / (1 + np.exp(-xs[0]))
        grad_logits = sigmoid_logits - xs[1]
        grad_logits = grad * grad_logits / xs[0].shape[0]
        yield grad_logits
        yield -grad_logits
        
    def __repr__(self) -> str: return "bce()"
        
        
def bce_with_logits_loss(logits, targets):
    return binary_cross_entropy_with_logits()(variable(logits), variable(targets))


class categorical_cross_entropy_with_integer_labels(operation):
    def forward(self, logits: Array, labels: Array) -> Array:
        max_logits = np.max(logits, axis=-1)[:, None]
        exp_logits = np.exp(logits - max_logits)
        sum_exp_logits = np.sum(exp_logits, axis=-1)[:, None]
        self.probs = exp_logits / sum_exp_logits
        log_probs = (logits - max_logits) - np.log(sum_exp_logits)
        loss = -np.mean(log_probs[np.arange(labels.shape[0]), labels])
        return loss
    
    def backward(self, xs: tuple[Array, ...], grad: Array) -> Iterable[Array]:
        self.probs[np.arange(xs[1].shape[0]), xs[1]] -= 1
        yield grad * self.probs / self.probs.size
    
    def __repr__(self) -> str: return "cce()"


def cce_loss(logits: variable, labels: variable):
    return categorical_cross_entropy_with_integer_labels()(variable(logits), variable(labels))


class sigmoid(operation):
    def forward(self, x: Array) -> Array:
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.sigmoid
    
    def backward(self, xs: tuple[Array, ...], grad: Array) -> Iterable[Array]:
        assert len(xs) == 1
        yield grad * self.sigmoid * (1 - self.sigmoid)
        
    def __repr__(self) -> str: return "sigmoid()"


class mse(operation):
    def forward(self, preds: Array, targets: Array):
        return np.mean((preds - targets) ** 2, axis=-1)
    
    def backward(self, xs, grad):
        grad = 2 * grad * (xs[0] - xs[1]) / xs[0].shape[0]
        yield grad
        yield -grad
        
    def __repr__(self) -> str: return "mse()"
        

def mse_loss(preds: variable, targets: variable):
    return mse()(variable(preds), variable(targets))


class relu(operation):
    def forward(self, x):
        return np.where(x > 0, x, 0)

    def backward(self, xs, grad):
        yield np.where(xs[0] > 0, grad, 0)
        
    def __repr__(self) -> str: return "relu()"