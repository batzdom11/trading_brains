"""
Import hook that redirects all `lightning.*` imports to `pytorch_lightning.*`.

pytorch-forecasting 1.6.1 internally uses `import lightning.pytorch` but the
`lightning` PyPI package is quarantined. We have `pytorch-lightning` installed
which provides the `pytorch_lightning` namespace. This hook bridges the gap.
"""
import importlib
import importlib.abc
import importlib.machinery
import sys
import types


class _LightningRedirectFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that redirects lightning.pytorch.* -> pytorch_lightning.*"""

    def find_module(self, fullname, path=None):
        if fullname == "lightning" or fullname.startswith("lightning."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        # Top-level 'lightning' package - create a namespace module
        if fullname == "lightning":
            mod = types.ModuleType("lightning")
            mod.__path__ = []
            mod.__package__ = "lightning"
            mod.__loader__ = self
            sys.modules["lightning"] = mod
            return mod

        # lightning.pytorch -> pytorch_lightning
        # lightning.pytorch.callbacks -> pytorch_lightning.callbacks
        # etc.
        if fullname == "lightning.pytorch" or fullname.startswith("lightning.pytorch."):
            real_name = fullname.replace("lightning.pytorch", "pytorch_lightning", 1)
        else:
            # For any other lightning.X, try pytorch_lightning.X
            real_name = fullname.replace("lightning.", "pytorch_lightning.", 1)

        try:
            real_mod = importlib.import_module(real_name)
        except ImportError:
            raise ImportError(f"No module named '{fullname}' (tried '{real_name}')")

        # Register under both names
        sys.modules[fullname] = real_mod
        
        # Set as attribute on parent
        parts = fullname.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr = parts
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, attr, real_mod)

        return real_mod


# Install the finder
sys.meta_path.insert(0, _LightningRedirectFinder())
