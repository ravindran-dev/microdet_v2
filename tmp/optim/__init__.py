import pkgutil
import importlib
from pathlib import Path

package_dir = Path(__file__).resolve().parent

for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
    if not module_name.startswith("__"):
        importlib.import_module(f"optim.{module_name}")

from .builder import build_optimizer

__all__ = ["build_optimizer"]
