import random
from micrograd.engine import Value

global _use_sumv
_use_sumv=False

# Enable to use the sumv optimization,
# instead of pairwise "+" (__add__)
def enable_sumv(enable):
    global _use_sumv
    _use_sumv = enable

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, id, nin, nonlin=True):
        self.id = id
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        global _use_sumv

        weighted = [wi*xi for wi,xi in zip(self.w, x)]

        act = (
            sum(weighted, self.b),
            Value.sumv(weighted) + self.b,
        )[_use_sumv]

        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron([{self.id}]{len(self.w)})"

class Layer(Module):

    def __init__(self, id, nin, nout, **kwargs):
        self.id = id
        self.neurons = [Neuron((id, i), nin, **kwargs) for i in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(i, sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
