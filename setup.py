"""Setup the package."""
from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import setup, find_packages


__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "needle.backend_ndarray.ndarray_backend_cpu",
        ["src/ndarray_backend_cpu.cc"],
        define_macros = [('VERSION_INFO', __version__)],
        cxx_std=11,
        language='c++'
    )
]
try:
    import torch
    assert torch.cuda.is_available()
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    build_ext = BuildExtension
    ext_modules.append(
        CUDAExtension(
            "needle.backend_ndarray.ndarray_backend_cuda",
            ["src/ndarray_backend_cuda.cu"],
            define_macros = [('VERSION_INFO', __version__)],
        )
    )
except Exception as e:
    pass

setup(
    name="needle",
    version=__version__,
    description="Deep Learning Framework in DLSys course",
    zip_safe=False,
    packages=find_packages("python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    setup_requires=["pybind11"],
    install_requires=["pybind11"],
    python_requires='>=3.8',
    url="dlsyscourse.org"
)
