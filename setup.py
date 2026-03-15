from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


ROOT = Path(__file__).resolve().parent


class build_ext(_build_ext):
    def run(self) -> None:
        super().run()
        runtime_dll = Path("D:/mingw64/bin/libwinpthread-1.dll")
        if runtime_dll.is_file():
            target = ROOT / "gomoku_ai" / runtime_dll.name
            shutil.copy2(runtime_dll, target)


ext_modules = [
    Extension(
        "gomoku_ai._cpp_backend",
        [str(ROOT / "cpp" / "gomoku_core.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++17", "/O2"] if sys.platform == "win32" else ["-std=c++17", "-O3"],
        extra_link_args=[] if sys.platform == "win32" else ["-static-libstdc++", "-static-libgcc"],
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
