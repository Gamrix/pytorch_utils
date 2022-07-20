#%%
import sympy
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map
import operator
from contextlib import contextmanager

aten = torch.ops.aten

# Brief explanation of the scheme below
sympy.init_printing()
#%%

meta_funcs = {}


def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f

        tree_map(add_func, op)
        return f

    return decorator


@register_meta([aten.add.Tensor])
def binary_func_meta(a, b, **kwargs):
    assert a.dim() == b.dim()
    for a_dim, b_dim in zip(a.shape, b.shape):
        if a_dim != 1:
            assert a_dim == b_dim or b_dim == 1
    return a


# Copied from prims
# todo: figure out how to unify
@register_meta(aten.cat.default)
def cat_meta(tensors, dim=0):
    concat_length = 0
    shape = tensors[0].shape
    for tensor in tensors:
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                concat_length = concat_length + length
            else:
                assert length == common_length
    new_shape = list(shape)
    new_shape[dim] = concat_length
    return tensors[0].new_empty(new_shape)


# PLACEHOLDER: Need real faketensor to propagate through actual meta functions/tensors
#############
# Thoughts:
# Note that for any operator that's *decomposed*, we don't actually need to explicitly propagate shapes for it! The shape guards should already be implicitly asserted by the decomposition itself.
# So, we're only left with operators that we don't have decompositions for (i.e. prims). For those, we can write meta functions. But, in fact, meta-tensors can *also* be thought of as decompositions - they're just decompositions which *only* match in terms of metadata (but not actual values).
# So, FakeTensor (which is what I'm using to propagate dynamic shapes without tracing) should call into decomps when possible, and call into meta funcs otherwise.
class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)

    def sizes(self):
        return self.shape

    def new_empty(self, shape):
        return FakeTensor(shape)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in meta_funcs:
            return meta_funcs[func](*args, **kwargs)
        raise RuntimeError(f"Unknown function {func}")


def propagate_meta(func, *args, **kwargs):
    def get_fake(x):
        return (
            FakeTensor(x.shape)
            if isinstance(x, torch.Tensor) or isinstance(x, ProxyTensor)
            else x
        )

    if func in meta_funcs:
        return meta_funcs[func](
            *tree_map(get_fake, args), **tree_map(get_fake, kwargs)
        ).shape
    raise RuntimeError(f"Unknown function {func}")


shape_funcs = {}


def shape_function(func, *maybe_args, **maybe_kwargs):
    if func in shape_funcs:
        return func(*maybe_args, maybe_kwargs)

    def make_tensor(x):
        if isinstance(x, list):
            return FakeTensor(x)
        return x

    args = tree_map(make_tensor, maybe_args)
    kwargs = tree_map(make_tensor, maybe_kwargs)
    out = propagate_meta(func, *args, **kwargs)
    return out.shape


from torch._C import _disabled_torch_function_impl


@contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        del guard


class ProxyTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, fake_elem, proxy, sym_shape, tracer):
        r = super().__new__(cls, fake_elem)  # type: ignore[call-arg]
        r.proxy = proxy
        r.sym_shape = sym_shape
        r.tracer = tracer
        return r

    __torch_function__ = _disabled_torch_function_impl

    @property
    def shape(self):
        return self.sym_shape

    @classmethod
    def __torch_dispatch__(cls, func_overload, types, args=(), kwargs=None):
        args = args if args else ()

        def get_proxy(x):
            return x.proxy if isinstance(x, ProxyTensor) else x

        kwargs = kwargs if kwargs else {}
        tracer = []

        def get_tracer(x):
            if isinstance(x, ProxyTensor):
                tracer.append(x.tracer)

        tree_map(get_tracer, (args, kwargs))
        assert len(tracer) > 0
        tracer = tracer[0]
        output_proxy = tracer.graph.call_function(
            func_overload, tree_map(get_proxy, args), tree_map(get_proxy, kwargs)
        )
        with no_dispatch():
            out = func_overload(*args, **kwargs)

        output_shape = propagate_meta(func_overload, *args, **kwargs)
        return ProxyTensor(out, output_proxy, output_shape, tracer)


# EXTENDABLE
class SymInt(object):
    def __init__(self, expr, tracer):
        self.expr = expr
        self.tracer = tracer

    def __bool__(self):
        return bool(self.tracer.evaluate_expr(self.expr))


magic_methods = {
    "add": lambda a, b: a + b,
    "radd": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "mod": lambda a, b: a % b,
    "eq": lambda a, b: sympy.Eq(a, b),
    "gt": lambda a, b: sympy.Gt(a, b),
    "lt": lambda a, b: sympy.Lt(a, b),
}

for method, func in magic_methods.items():
    method_name = f"__{method}__"

    def create_magic_impl(func):
        def magic_impl(self, other):
            if isinstance(other, SymInt):
                other = other.expr
            return SymInt(func(self.expr, other), self.tracer)

        return magic_impl

    setattr(SymInt, method_name, create_magic_impl(func))

# EXTENDABLE
class Tracer(object):
    def __init__(self):
        self.graph = fx.Graph()
        self.guards = []
        self.shape_env = {}

    def create_symbol(self, name, val, shape_env=None):
        if shape_env is None:
            shape_env = self.shape_env
        sym_int = sympy.symbols(name, integer=True)
        shape_env[sym_int] = val
        return SymInt(sym_int, self)

    def evaluate_expr(self, expr):
        concrete_val = expr.subs(self.shape_env)
        self.guards.append((expr, concrete_val))
        return concrete_val

    def create_args(self, *args):
        proxy_args = []
        for idx, arg in enumerate(args):
            name = chr(idx + 65)
            proxy = self.graph.placeholder(name)
            sym_shapes = [
                self.create_symbol(f"{name}_{idx}", i)
                for idx, i in enumerate(arg.shape)
            ]
            proxy_args.append(ProxyTensor(arg, proxy, sym_shapes, self))
        return proxy_args

    def evaluate_guards(self, *args):
        env = {}
        # args = [create_proxy(chr(idx + 65), i.shape) for idx, i in enumerate(args)]
        for idx, arg in enumerate(args):
            name = chr(idx + 65)
            sym_shapes = [
                self.create_symbol(f"{name}_{idx}", i, env)
                for idx, i in enumerate(arg.shape)
            ]
        return all(guard.subs(env) == value for guard, value in self.guards)


def dynamic_trace(f, args):
    tracer = Tracer()
    args = tracer.create_args(*args)
    f(*args)
    return tracer


## Start reading from here!
# Basically, we create symbolic variables for the shapes of each of our variables
# Then, we simply trace like we would normally. However, our proxies now also contain "symbolic shapes" on them.
# So, when we perform any operation, the symbolic shape must be propagated to the output tensor.
# At any point, we have an expression for the shape of any tensor in terms of the input tensors
# When we come to control flow (including asserts), we do 2 things:
# 1. Evaluate the control flow expression to a boolean. Remember that any tensor's shape can be evaluated in terms of the input tensors!
# 2. Store a guard for our cache. This guard allows us to check if the trace is valid at the input!


def f(a, b):
    c = torch.add(a, b)
    if c.shape[0] > 3:
        return torch.add(c, c)
    else:
        return torch.sub(c, c)


tracer = dynamic_trace(f, [torch.rand(4), torch.rand(4)])
print(tracer.graph)
print(tracer.guards)
assert tracer.evaluate_guards(torch.rand(6), torch.rand(6))
assert not tracer.evaluate_guards(torch.rand(3), torch.rand(3))


def f2(a, b):
    c = torch.cat([a, b])
    if c.shape[0] % 2 == 0:
        return torch.add(c, c)
    else:
        return torch.sub(c, c)


tracer = dynamic_trace(f2, [torch.randn(3), torch.randn(5)])
print(tracer.graph)
# It's checking whether a.shape[0] + b.shape[0] is divisible by 2
print(tracer.guards)  # [(Eq(Mod(A_0 + B_0, 2), 0), True)
# It's valid because 4 + 4 % 2 == 0
assert tracer.evaluate_guards(torch.rand(4), torch.rand(4))
# It's not valid because 3 + 4 % 2 != 0
assert not tracer.evaluate_guards(torch.rand(3), torch.rand(4))

########################
# Extending Tracer
########################


class TracerDouble(Tracer):
    def __init__(self):
        self.graph = fx.Graph()
        self.guards = []
        self.shape_env = [{}, {}]

    def create_symbol(self, name, vals, shape_env=None):
        if shape_env is None:
            shape_env = self.shape_env

        sym_int = sympy.symbols(name, integer=True)
        for idx, val in enumerate(vals):
            shape_env[idx][sym_int] = val
        return SymInt(sym_int, self)

    def evaluate_expr(self, expr):
        concrete_vals = [expr.subs(env) for env in self.shape_env]
        if concrete_vals[0] != concrete_vals[1]:
            raise RuntimeError(
                "Not all provided values follow the same shape-dependent control flow!"
            )
        assert concrete_vals[0] == concrete_vals[1]
        concrete_val = concrete_vals[0]
        self.guards.append((expr, concrete_val))
        return concrete_val

    def create_args(self, *args):
        proxy_args = []
        for idx, arg in enumerate(args):
            name = chr(idx + 65)
            proxy = self.graph.placeholder(name)
            sym_shapes = [
                self.create_symbol(f"{name}_{idx}", i)
                for idx, i in enumerate(zip(arg[0].shape, arg[1].shape))
            ]
            proxy_args.append(ProxyTensor(arg[0], proxy, sym_shapes, self))
        return proxy_args

    def evaluate_guards(self, *args):
        env = {}
        # args = [create_proxy(chr(idx + 65), i.shape) for idx, i in enumerate(args)]
        for idx, arg in enumerate(args):
            name = chr(idx + 65)
            sym_shapes = [
                self.create_symbol(f"{name}_{idx}", i, env)
                for idx, i in enumerate(arg.shape)
            ]
        return all(guard.subs(env) == value for guard, value in self.guards)


def dynamic_trace_double(f, args):
    tracer = TracerDouble()
    args = tracer.create_args(*args)
    f(*args)
    return tracer


def f(x):
    if x.shape[0] > 3:
        return x + x
    else:
        return x


# Works!
tracer = dynamic_trace_double(f, [(torch.randn(4), torch.randn(5))])
# Errors!

# dynamic_trace_double(f, [(torch.randn(2), torch.randn(4))])
print(tracer.guards)
#%%

print(type(tracer.guards[0][0]))

#%%

########################
# Further points
########################
##### Nonzero ######
#  For nonzero, things start to get more complicated. Certain cases are ... infeasible (i.e. force a graph break). For example
def f(x):
    y = x.nonzero()
    if y.shape[0] % 2 == 0:
        return y.cos()
    else:
        return y.sin()


# However, there are other cases that *could* work. For example
def f(x):
    y = x.nonzero()
    val = torch.sum(x[y])
    if val.shape[0] > 3:
        return val.cos()
    else:
        return val.sin()


# To handle this, our shape propagating FakeTensor needs to return a fresh symbolic dim when we call nonzero. Certain backends (such as XLA) may want this fresh symbolic dim to have additional constraints on it.
# Then, if we try to evaluate a concrete value for a shape (and it involves this fresh dim), then we error.
# Note that there is some latitude for more advanced solving schemes - for example, perhaps the upper-bound constraint already rules out the conditional we are trying to evaluate.


##### "Optimizer-only" shape guards ######
# It's plausible that backends may want to introduce "optimizer-only" shape-guards. Here are two examples
def efficient_decomp(x):
    pass


def decomp(x):
    pass


def adaptive_pool(x, size):
    if x.shape[0] % size == 0:
        return efficient_decomp(x)
    else:
        return decomp(x)


def f(a, b, c, d):
    return [adaptive_pool(x, 3) for x in [a, b, c, d]]


# Any time we're introducing guards, we can potentially have exponential blowup. So, in this case, we probably want some way of denoting that a guard is "optimization-only", and have some way of falling back on it if we're triggering it too much (or say, we triggered the False case).

# Another example is for stuff like this (taken from inductor). In our decompositions, we don't want to specialize on x_size. But, backends may want it for
def split(x, sizes, dim):
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.guard_static_shape(x.get_size()[dim])
    if isinstance(sizes, int):
        sizes = [sizes] * ((x_size + sizes - 1) // sizes)
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(slice_(x, dim, start, end))
        start = end
    return result


def f(a, b):
    assert a.shape[0] == b.shape[0]
    return torch.cat([a, b])


# %%
