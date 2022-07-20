#%%
from regex import W
from sqlalchemy import Integer
import sympy

# NOTES TO READERS:
# This is an exploration of using Sympy to trace through the SSA code
# However, upon making this, I realized that Sympy has no way to
# evaluate systems of inequalities, and therefore including Sympy into
# this has no benefits. Therefore, I instead wrote the
# upper_bound_symint_mock.py which removes uses of Sympy.

# Sample symint
class SymInt(object):
    def __init__(self, expr, tracer):
        self.expr = expr
        self.tracer = tracer

    def __bool__(self):
        return bool(self.tracer.evaluate_expr(self.expr))

magic_methods = {
    'add': lambda a, b: a + b,
    'radd': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': lambda a, b: a / b,
    'mod': lambda a, b: a % b,
    'eq': lambda a, b: sympy.Eq(a, b),
    'gt': lambda a, b: sympy.Gt(a, b),
    'lt': lambda a, b: sympy.Lt(a, b),
}

for method, func in magic_methods.items():
    method_name = f'__{method}__'
    def create_magic_impl(func):
        def magic_impl(self, other):
            if isinstance(other, SymInt):
                other = other.expr
            return SymInt(func(self.expr, other), self.tracer)
        return magic_impl

    setattr(SymInt, method_name, create_magic_impl(func))

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
            name = chr(idx+65)
            proxy = self.graph.placeholder(name)
            sym_shapes = [self.create_symbol(f'{name}_{idx}', i) for idx, i in enumerate(arg.shape)]
            proxy_args.append(ProxyTensor(arg, proxy, sym_shapes, self))
        return proxy_args

    def evaluate_guards(self, *args):
        env = {}
        # args = [create_proxy(chr(idx + 65), i.shape) for idx, i in enumerate(args)]
        for idx, arg in enumerate(args):
            name = chr(idx+65)
            sym_shapes = [self.create_symbol(f'{name}_{idx}', i, env) for idx, i in enumerate(arg.shape)]
        return all(guard.subs(env) == value for guard, value in self.guards)


#%% Scratch for Sympy testing
import sympy
input_a = sympy.symbols('a', integer=True)
input_b = sympy.symbols('b', integer=True)

output_a = sympy.symbols('out_a', integer=True)

# Maybe this feature is not part of Sympy
# https://stackoverflow.com/questions/17048180/how-to-solve-multivariate-inequalities-with-python-and-sympy
# Looking at all the inequality solvers in Sympy, there is no support for this
# https://docs.sympy.org/dev/modules/solvers/inequalities.html

sympy_exprs = [sympy.Lt(input_a, 5), sympy.Eq(output_a, input_a + 2)]
from sympy.solvers.inequalities import reduce_rational_inequalities
# print(reduce_rational_inequalities([sympy_exprs], output_a))
# print(sympy.linsolve(sympy_exprs, output_a))
# print(reduce_rational_inequalities([[input_a**2 <= 0], [sympy.Eq(output_a, input_a + 2)]], output_a))


# Sympy doesn't have a built-in way to solve systems of inequalities
# Therefore, we will have to build a poor man's inequality solver

# What constraints can we apply to the inequlities so that
# We treat all symbolic values as having an upper bound
# and is_dynamic

# We have a system of inequalities
# 1. start with a mapping of variable names to their lower, upper bounds
# 2. Iterate over the inequalities and apply the mapping to the inequalities
# Do this in a loop until we hit a fixed point
help(sympy_exprs[0])


# Build tensors with SymInt objects
INT_MAX = 2**62
import dataclasses

@dataclasses.dataclass(frozen=True)
class IntegerSymInt(object):
    lower: int = -INT_MAX
    # Upper is also a concrete value for tracing
    upper: int = INT_MAX

    @staticmethod
    def new_tensor_size(max=INT_MAX):
        return IntValueBounds(lower=1, upper=max)

    def __bool__(self):
        return bool(self.upper)

    def __add__(self, other):
        if isinstance(other, IntegerSymInt):
            return IntegerSymInt(lower=self.lower + other.lower, upper=self.upper + other.upper)
        return IntegerSymInt(lower=self.lower + other, upper=self.upper + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, IntegerSymInt):
            return IntegerSymInt(lower=self.lower - other.upper, upper=self.upper - other.lower)
        return IntegerSymInt(lower=self.lower - other, upper=self.upper - other)

    def __rsub__(self, other):
        # I do not expect this to be needed for the shape tracing that we currently need
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, IntegerSymInt):
            return IntegerSymInt(lower=self.lower * other.lower, upper=self.upper * other.upper)
        return IntegerSymInt(lower=self.lower * other, upper=self.upper * other)

    def __mod__(self, other):
        assert(isinstance(other, int))
        return IntegerSymInt(lower=0, upper=other)

    def __idiv__(self, other):
        assert(isinstance(other, int))


    # Comparisons:
    # We will likely need to figure out which in which ops these
    # operations are used for comparisons (and not assertions)
    # and manually figure out is_dynamic information for them
    def __gt__(self, other):
        assert(isinstance(other, int))
        return self.upper > other

    def __ge__(self, other):
        assert(isinstance(other, int))
        return self.upper >= other

    def __lt__(self, other):
        assert(isinstance(other, int))
        return self.upper < other

    def __le__(self, other):
        assert(isinstance(other, int))
        return self.upper <= other


    # Equality: we are generally
    def __eq__(self, other):
        assert(isinstance(other, int))
        assert(other < 2)
        return False


class SimpleStaticTracer(Tracer):
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
            for dim_size, is_dyn in zip(arg.shape(), dims_dynamic):
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
            dyn_info.append(arg.shape)
        return dyn_info



# Run Tensors through a shape function to get out sympy results
# See this diff for how we would collect
# Sympy shape propagation symbols from Shape Functions
# https://github.com/pytorch/pytorch/pull/81093/files

# What kind of operations can we expect:
# Assertions that a shape has a linear conditional relationship to a integer value
# Assertions that a shape has a linear conditional relationship to another shape
#      For the first two, we can ignore, as we know that these conditionals will hold
#      for all inputs?
# The problem is what would happen with stuff like slice, where the tensor is dynamic
# if start_val > self[dim]:
#     start_val = self[dim]
#  elif end_val >= self[dim]:
#     end_val = self[dim]
# Slice([5], dim=0, 2, 3)
#
# Assignment of inputs shapes to intermediary shapes through operations.

simpy.lambdify([input_a], sympy.Eq(output_a, input_a + 2))




# Deduce shapes from the Sympy results

# %%
