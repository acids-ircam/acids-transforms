"""Tracing backends for data-dependent algorithms (PGHI, sine bank) and helpers (frame).

These algorithms loop and branch on values, which make_fx (and torch.export / torch.compile) cannot trace. Each of
them has :

- an exact implementation, used in eager mode and by TorchScript (scripted modules run it in libtorch / nn~) ;
- a custom operator wrapping the exact implementation (torch.library.custom_op), recorded by make_fx as a single
  opaque node : traced graphs then need acids_transforms to be imported to run ;
- a pure implementation, made of plain aten operations without data-dependent control flow (approximate for PGHI and
  the sine bank), that make_fx traces into portable graphs.

While tracing, the "custom_op" backend uses the operators, and the "pure" backend the pure implementations. The
backend is "custom_op" if torch.library.custom_op is available (torch >= 2.4), and "pure" otherwise ; set_backend
changes it. Note that torch.fx.wrap has no effect here : it only applies to torch.fx.symbolic_trace, not to make_fx.
"""
from typing import Callable, Optional
import torch

__all__ = ["BACKENDS", "get_backend", "set_backend", "opaque_op", "is_fx_tracing", "select_implementation"]

BACKENDS = ("custom_op", "pure")
HAS_CUSTOM_OPS = hasattr(torch, "library") and hasattr(torch.library, "custom_op")
_backend = "custom_op" if HAS_CUSTOM_OPS else "pure"


def get_backend() -> str:
    """tracing backend : "custom_op" or "pure"."""
    return _backend


def set_backend(backend: str) -> None:
    """sets the tracing backend : "custom_op" (opaque operators, needs torch >= 2.4) or "pure" (plain aten operations,
    approximate PGHI and sine bank)."""
    global _backend
    if backend not in BACKENDS:
        raise ValueError("backend must be in %s, got %s" % (BACKENDS, backend))
    if backend == "custom_op" and not HAS_CUSTOM_OPS:
        raise RuntimeError("custom operators need torch.library.custom_op (torch >= 2.4)")
    _backend = backend


def opaque_op(name: str, fake: Callable) -> Callable:
    """decorator registering fn as the custom operator name ("namespace::op"), traced by make_fx as a single node.
    ``fake(*args)`` must return empty tensors with the shapes and dtypes of the outputs. Returns None if custom operators
    are not available."""
    def decorator(fn: Callable) -> Optional[Callable]:
        if not HAS_CUSTOM_OPS:
            return None
        try:
            op = torch.library.custom_op(name, mutates_args=())(fn)
        except RuntimeError:
            # already registered (e.g. module reloaded)
            namespace, op_name = name.split("::")
            return getattr(getattr(torch.ops, namespace), op_name)
        op.register_fake(fake)
        return op
    return decorator


@torch.jit.unused
def is_fx_tracing() -> bool:
    """True inside make_fx, torch.export or torch.compile. Modules should then not store traced tensors in their
    buffers (they would leak in later calls)."""
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling") and compiler.is_compiling():
        return True
    try:
        from torch.fx.experimental.proxy_tensor import get_proxy_mode
    except ImportError:
        return False
    return get_proxy_mode() is not None


@torch.jit.unused
def select_implementation(exact: Callable, pure: Callable, op: Optional[Callable]) -> Callable:
    """implementation to call in eager mode : exact outside tracing ; while tracing, op with the "custom_op" backend,
    pure otherwise."""
    if not is_fx_tracing():
        return exact
    if _backend == "custom_op" and op is not None:
        return op
    return pure
