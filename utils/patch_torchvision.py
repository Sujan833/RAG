import os
import sys
import types
from importlib.machinery import ModuleSpec

# Lock CPU threads to 1 on Windows to prevent OpenMP/PyTorch crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class TorchvisionDummyModule(types.ModuleType):
    """Synthetic dummy module for torchvision that satisfies importlib, transformers, and Streamlit watcher."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []
        self.__file__ = "dummy_torchvision"
        self.__loader__ = None
        self.__spec__ = ModuleSpec(name, None, is_package=True)

    def __getattr__(self, item: str):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        sub_name = f"{self.__name__}.{item}"
        if sub_name not in sys.modules:
            mod = TorchvisionDummyModule(sub_name)
            sys.modules[sub_name] = mod
            setattr(self, item, mod)
        return sys.modules[sub_name]

    def __call__(self, *args, **kwargs):
        return None


class TorchvisionLoader:
    def create_module(self, spec):
        name = spec.name
        if name not in sys.modules:
            sys.modules[name] = TorchvisionDummyModule(name)
        return sys.modules[name]

    def exec_module(self, module):
        pass


class TorchvisionFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            return ModuleSpec(fullname, TorchvisionLoader(), is_package=True)
        return None


def apply_torchvision_patch():
    """Intercept all torchvision imports to prevent Streamlit watcher & transformers C++ errors cleanly."""
    if not any(isinstance(f, TorchvisionFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, TorchvisionFinder())

    # Pre-populate torchvision and submodules in sys.modules
    tv = TorchvisionDummyModule("torchvision")
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = TorchvisionDummyModule("torchvision.transforms")
    sys.modules["torchvision.transforms.v2"] = TorchvisionDummyModule("torchvision.transforms.v2")
    sys.modules["torchvision.transforms.v2.functional"] = TorchvisionDummyModule("torchvision.transforms.v2.functional")
    sys.modules["torchvision.io"] = TorchvisionDummyModule("torchvision.io")

    # Patch streamlit local_sources_watcher to prevent torchvision tracebacks
    try:
        import streamlit.watcher.local_sources_watcher as watcher

        if hasattr(watcher, "get_module_paths"):
            orig_get_module_paths = watcher.get_module_paths

            def safe_get_module_paths(module):
                mod_name = getattr(module, "__name__", "")
                if mod_name and mod_name.startswith("torchvision"):
                    return set()
                try:
                    return orig_get_module_paths(module)
                except Exception:
                    return set()

            watcher.get_module_paths = safe_get_module_paths
    except Exception:
        pass


apply_torchvision_patch()
