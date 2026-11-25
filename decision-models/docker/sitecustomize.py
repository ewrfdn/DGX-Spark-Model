"""Point CPython's include path at the headers we extracted from python3.12-dev.

The target box has no python3.12-dev package installed; Triton needs Python.h to build its
CUDA driver shim, so we ship the headers under /tmp/pydev and redirect sysconfig here.
"""
import os
import sysconfig

_PREFIX = os.environ.get("PY_HEADERS", "/tmp/pydev/usr/include/python3.12")
if os.path.isdir(_PREFIX):
    _orig_get_paths = sysconfig.get_paths
    _orig_get_path = sysconfig.get_path

    def get_paths(*args, **kwargs):
        paths = _orig_get_paths(*args, **kwargs)
        paths["include"] = _PREFIX
        return paths

    def get_path(name, *args, **kwargs):
        if name == "include":
            return _PREFIX
        return _orig_get_path(name, *args, **kwargs)

    sysconfig.get_paths = get_paths
    sysconfig.get_path = get_path
