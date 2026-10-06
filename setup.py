"""Build configuration for irvine-traffic-ai.

Compiles the native routing core in cpp_engine/ into a pybind11 extension
module importable as `traffic_engine`.

Build:
    pip install pybind11 setuptools
    pip install -e .
"""

from glob import glob
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

HERE = Path(__file__).parent
CPP_ENGINE = HERE / "cpp_engine"

# Every .cpp under cpp_engine/src is compiled into the extension. Sorted so
# the source order is deterministic across platforms (ccache/reproducibility).
SOURCES = sorted(glob(str(CPP_ENGINE / "src" / "*.cpp")))

ext_modules = [
    Pybind11Extension(
        "traffic_engine",
        SOURCES,
        include_dirs=[str(CPP_ENGINE / "include")],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"0.1.0"')],
    ),
]

setup(
    name="irvine-traffic-ai",
    version="0.1.0",
    description="Traffic prediction and routing engine for Irvine, CA",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*", "app", "app.*"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
