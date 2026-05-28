from __future__ import annotations

import sys
import shutil
from pathlib import Path

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


ROOT = Path(__file__).resolve().parent
BUILD_ROOT = ROOT / "outputs" / "build"
INTERMEDIATE_ROOT = BUILD_ROOT / "intermediate"


class build_ext(_build_ext):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.build_temp = str(BUILD_ROOT / "temp")
        self.build_lib = str(BUILD_ROOT / "lib")

    def run(self) -> None:
        BUILD_ROOT.mkdir(parents=True, exist_ok=True)
        INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)
        super().run()
        runtime_dll = Path("D:/mingw64/bin/libwinpthread-1.dll")
        if runtime_dll.is_file():
            target = ROOT / "gomoku_ai" / runtime_dll.name
            shutil.copy2(runtime_dll, target)
        self._relocate_msvc_artifacts()

    def _relocate_msvc_artifacts(self) -> None:
        patterns = ("*.obj", "*.exp", "*.lib")
        for pattern in patterns:
            for artifact in ROOT.glob(f"cpp/**/{pattern}"):
                target = INTERMEDIATE_ROOT / artifact.name
                if target.exists():
                    target.unlink()
                shutil.move(str(artifact), str(target))


ext_modules = [
    Extension(
        "gomoku_ai._cpp_backend",
        [
            str(ROOT / "cpp" / "RewardEvaluator.cpp"),
            str(ROOT / "cpp" / "GameStateStore.cpp"),
            str(ROOT / "cpp" / "RewardConfigStore.cpp"),
            str(ROOT / "cpp" / "StateValueRegistry.cpp"),
            str(ROOT / "cpp" / "direction_encoding.cpp"),
            str(ROOT / "cpp" / "utils" / "direction_pattern_lookup.cpp"),
            str(ROOT / "cpp" / "precompute" / "DirectionDeltaTable.cpp"),
        ],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++17", "/O2", "/utf-8"] if sys.platform == "win32" else ["-std=c++17", "-O3"],
        extra_link_args=[] if sys.platform == "win32" else ["-static-libstdc++", "-static-libgcc"],
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    options={"build": {"build_base": str(BUILD_ROOT)}},
)
