# tmp/model/__init__.py
import importlib
import pkgutil
import warnings
from .registry import MODEL_REGISTRY, register_module, get_module, list_registered
from . import backbone, neck, head, module

_subpkgs = ("backbone", "head", "neck", "module")

for sub in _subpkgs:
    pkgname = f"tmp.model.{sub}"
    try:
        pkg = importlib.import_module(pkgname)
    except Exception as e:
        warnings.warn(f"tmp.model: could not import '{pkgname}' at package init: {e}")
        continue

    # import all modules in the subpackage so registration decorators execute
    try:
        for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            modname = f"{pkgname}.{name}"
            try:
                importlib.import_module(modname)
            except Exception as e:
                warnings.warn(f"Failed importing {modname}: {e}")
    except Exception:
        # some subpackages may not be real packages (no __path__), ignore
        pass

__all__ = [
    "MODEL_REGISTRY",
    "register_module",
    "get_module",
    "list_registered",
]
