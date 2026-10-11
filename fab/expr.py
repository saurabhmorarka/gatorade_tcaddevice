"""Safe parameter expressions for process YAML: "Lg/2 + spacer", "30nm",
"0.05um", "max(W, 20)". Lengths are in nm; a number with a unit suffix
(nm, um, A for angstrom) is converted to nm. Only arithmetic, parameter
names and a few math functions are allowed (parsed with ast, never eval).
"""
import ast
import math
import operator
import re

_UNITS = {"nm": 1.0, "um": 1.0e3, "A": 0.1}
_NUM_UNIT = re.compile(r"(?<![\w.])((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(nm|um|A)\b")
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod}
_FUNCS = {"min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "exp": math.exp, "log10": math.log10}


class ExprError(ValueError):
    pass


def _strip_units(text):
    return _NUM_UNIT.sub(lambda m: repr(float(m.group(1)) * _UNITS[m.group(2)]), text)


def evaluate(value, params=None):
    """A number, or an expression string over `params`, as a float."""
    if isinstance(value, bool):
        raise ExprError(f"expected a number, got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ExprError(f"expected a number or expression, got {value!r}")
    params = params or {}
    try:
        tree = ast.parse(_strip_units(value.strip()), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"bad expression {value!r}: {e.msg}") from None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id not in params:
                raise ExprError(f"unknown parameter {n.id!r} in {value!r}")
            return evaluate(params[n.id], params)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise ExprError(f"not allowed in {value!r}: {ast.dump(n)[:60]}")

    return ev(tree)


def resolve_params(params):
    """Evaluate every parameter (they may refer to each other); non-numeric
    values (dicts, lists, names of materials) are kept as they are."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
            try:
                out[k] = evaluate(v, params)
                continue
            except ExprError:
                # a single word that is not a parameter: a name (a material,
                # "SiO2", "SiGe0.3"), not an expression
                if isinstance(v, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", v.strip()) \
                        and v.strip() not in params:
                    out[k] = v
                    continue
                raise
        out[k] = v
    return out
