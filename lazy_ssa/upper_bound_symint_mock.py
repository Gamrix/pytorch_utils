# %%
import torch
import torch.fx as fx
from torch.utils._pytree import tree_map
from contextlib import contextmanager
from torch._C import _disabled_torch_function_impl
import dataclasses

aten = torch.ops.aten

# Brief explanation of the scheme below

meta_funcs = {}


def register_meta(op):
    def decorator(f):
        def add_func(op):
            meta_funcs[op] = f

        tree_map(add_func, op)
        return f

    return decorator


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

# NEW Stuff for static Tracing


@register_meta([aten.add.Tensor])
def binary_func_meta(a, b, **kwargs):
    assert a.dim() == b.dim()
    new_shape = []
    for a_dim, b_dim in zip(a.shape, b.shape):
        if a_dim != 1:
            assert a_dim == b_dim or b_dim == 1
            new_shape.append(a_dim)
        else:
            new_shape.append(b_dim)

    return FakeTensor(new_shape)



INT_MAX = 2**62


@dataclasses.dataclass()
class IntegerSymInt(object):
    lower: int = -INT_MAX
    # Upper is also a concrete value for tracing
    upper: int = INT_MAX

    def __post_init__(self):
        assert isinstance(self.lower, int)
        assert isinstance(self.upper, int)

    @staticmethod
    def new_tensor_size(max=INT_MAX):
        return IntegerSymInt(lower=1, upper=max)

    def __bool__(self):
        return bool(self.upper)

    def __add__(self, other):
        if isinstance(other, IntegerSymInt):
            return IntegerSymInt(
                lower=self.lower + other.lower, upper=self.upper + other.upper
            )
        return IntegerSymInt(lower=self.lower + other, upper=self.upper + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, IntegerSymInt):
            return IntegerSymInt(
                lower=self.lower - other.upper, upper=self.upper - other.lower
            )
        return IntegerSymInt(lower=self.lower - other, upper=self.upper - other)

    def __rsub__(self, other):
        # I do not expect this to be needed for the shape functions that we currently need
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, IntegerSymInt):
            return IntegerSymInt(
                lower=self.lower * other.lower, upper=self.upper * other.upper
            )
        return IntegerSymInt(lower=self.lower * other, upper=self.upper * other)

    def __mod__(self, other):
        assert isinstance(other, int)
        return IntegerSymInt(lower=0, upper=other)

    def __idiv__(self, other):
        assert isinstance(other, int)

    # Comparisons:
    # We will likely need to figure out which in which ops these
    # operations are used for comparisons (and not assertions)
    # and manually figure out is_dynamic information for them
    def __gt__(self, other):
        assert isinstance(other, int)
        return self.upper > other

    def __ge__(self, other):
        assert isinstance(other, int)
        return self.upper >= other

    def __lt__(self, other):
        assert isinstance(other, int)
        return self.upper < other

    def __le__(self, other):
        assert isinstance(other, int)
        return self.upper <= other

    def __eq__(self, other):
        # Equality: There have to be some hacks
        assert isinstance(other, int)
        if self.upper == other:
            self.lower = other
            return True
        assert other < 2
        return False

    def is_dynamic(self):
        assert self.upper >= self.lower
        return self.upper != self.lower


class SimpleStaticTracer(object):
    def __init__(self):
        self.graph = fx.Graph()

    def create_static_args(self, args, is_dynamic_data):
        # is_dynamic_data: List[Optional[List[bool]]]
        # for nontensor inputs, this is None
        # For Tensor inputs, it marks which dims are dynamic
        proxy_args = []
        for arg_index, (arg, dims_dynamic) in enumerate(zip(args, is_dynamic_data)):
            if dims_dynamic is None:
                # Not a Tensor
                proxy_args.append(arg)
                continue

            name = chr(arg_index + 65)
            proxy_name = self.graph.placeholder(name)
            sym_shapes = []
            for dim_size, is_dyn in zip(arg.shape, dims_dynamic):
                if is_dyn:
                    sym_shapes.append(IntegerSymInt.new_tensor_size(dim_size))
                else:
                    sym_shapes.append(dim_size)
            proxy_args.append(ProxyTensor(arg, proxy_name, sym_shapes, self))
        return proxy_args

    def get_return_dyn_info(self, *args):
        dyn_info = []
        for arg in args:
            if not isinstance(arg, ProxyTensor):
                dyn_info.append(None)
                continue
            shape = arg.shape
            new_shape = []
            for dim in shape:
                if isinstance(dim, IntegerSymInt) and not dim.is_dynamic():
                    new_shape.append(dim.upper)
                else:
                    new_shape.append(dim)
            dyn_info.append(new_shape)
        return dyn_info


def static_trace(f, args, is_dynamic):
    tracer = SimpleStaticTracer()
    args = tracer.create_static_args(args, is_dynamic)

    # we would find a way to call the metafunction more directly
    # but this would do as a prototype
    fn_results = f(*args)

    return tracer.get_return_dyn_info(fn_results)


# Fake meta functions to exercise the various operations that
# we expect a SymInt to support.
@register_meta([aten.sub.Tensor])
def test_subtraction(a, b):
    assert a.dim() == b.dim()
    new_shape = list(a.shape)
    new_shape[0] = a.shape[0] - b.shape[0]
    return a.new_empty(new_shape)


@register_meta([aten.relu.default])
def test_inequality_with_const(a):
    new_shape = list(a.shape)
    assert new_shape[0] < 5
    return a.new_empty(new_shape)


@register_meta([aten.mul.Tensor])
def test_equality(a, b):
    new_shape = list(a.shape)
    assert new_shape[0] == b.shape[0]
    return a.new_empty(new_shape)


# Demonstrate correctness on the cat_meta
# Also demonstrate correctness on the various other fake functions

def f(a, b):
    return torch.add(a, b)


dyn_info = static_trace(f, [torch.rand(4, 4, 4), torch.rand(4, 4, 1)], [[True, False, True], [False, False, False]])
print(dyn_info)
# Returns: [[4, 4, IntegerSymInt(lower=1, upper=4)]]
dyn_info = static_trace(f, [torch.rand(4, 4, 1), torch.rand(4, 4, 4)], [[False, False, False], [True, False, True]])
print(dyn_info)
# Returns: [[4, 4, IntegerSymInt(lower=1, upper=4)]]

dyn_info = static_trace(torch.relu, [torch.rand(4, 4, 1)], [[True, False, True]])
print(dyn_info)
# Returns [IntegerSymInt(lower=1, upper=4), 4, 1]]
# 1 is not a symint because we know that to be constant.

dyn_info = static_trace(torch.sub, [torch.rand(8), torch.rand(1)], [[True], [False]])
print(dyn_info)


# %%
