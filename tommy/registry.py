"""Tool registry: the only way the model can make anything happen.

A tool is a plain Python function decorated with @tool. The decorator records its name,
risk tier and a JSON schema built from the type hints and docstring, so adding a tool is one
function and nothing else. The model's output is only ever looked up here and type-checked;
it is never executed as text.
"""
import inspect, typing
from dataclasses import dataclass
from typing import Callable

RISKS = ("safe", "confirm")
_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class Reply:
    """What Jarvis answers: the text for the bubble, and its kind, which picks the spoken line (make_greetings.py)."""
    text: str
    kind: str = "done"      # done info cancelled declined not_understood no_app not_open no_time failed locked


@dataclass
class Tool:
    name: str
    fn: Callable
    risk: str
    description: str
    params: dict          # name -> {"type":..., "description":..., "enum"?:[...]}
    required: list
    says: str = "done"    # kind of Reply when the tool returns plain text

    def schema(self):
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.params, "required": self.required}}


TOOLS: dict[str, Tool] = {}


def tool(risk="safe", says="done", **param_docs):
    """@tool(risk="confirm", after="a time span such as '25 minutes'")"""
    assert risk in RISKS

    def wrap(fn):
        hints, sig = typing.get_type_hints(fn), inspect.signature(fn)
        params, required = {}, []
        for name, p in sig.parameters.items():
            t = hints.get(name, str)
            entry = {"type": _PY_TO_JSON.get(t, "string")}
            if typing.get_origin(t) is typing.Literal:
                entry = {"type": "string", "enum": list(typing.get_args(t))}
            if name in param_docs:
                entry["description"] = param_docs[name]
            params[name] = entry
            if p.default is inspect.Parameter.empty:
                required.append(name)
        doc = (inspect.getdoc(fn) or "").split("\n")[0]
        TOOLS[fn.__name__] = Tool(fn.__name__, fn, risk, doc, params, required, says)
        return fn
    return wrap


def check_args(name: str, args: dict):
    """Validate a model-proposed call. Returns (tool, cleaned_args) or raises ValueError."""
    t = TOOLS.get(name)
    if t is None:
        raise ValueError(f"unknown tool {name!r}")
    out = {}
    for key, val in args.items():
        spec = t.params.get(key)
        if spec is None:
            raise ValueError(f"{name}: unexpected argument {key!r}")
        want = spec["type"]
        if want == "integer" and not isinstance(val, bool):
            try: val = int(val)
            except (TypeError, ValueError): raise ValueError(f"{name}.{key}: not an integer")
        elif want == "number":
            try: val = float(val)
            except (TypeError, ValueError): raise ValueError(f"{name}.{key}: not a number")
        elif want == "boolean" and not isinstance(val, bool):
            raise ValueError(f"{name}.{key}: not a boolean")
        elif want == "string":
            val = str(val)
            if "enum" in spec and val not in spec["enum"]:
                raise ValueError(f"{name}.{key}: {val!r} not in {spec['enum']}")
        out[key] = val
    missing = [r for r in t.required if r not in out]
    if missing:
        raise ValueError(f"{name}: missing {missing}")
    return t, out


def load_all():
    """Import every module in tommy/tools so their @tool decorators run."""
    import importlib, pkgutil
    from . import tools
    for m in pkgutil.iter_modules(tools.__path__):
        importlib.import_module(f"{tools.__name__}.{m.name}")
    return TOOLS
