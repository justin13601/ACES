"""PROTOTYPE (not ready to merge): builds the optional native pass for MEDS plain predicates (aces._native).

Package metadata stays in pyproject.toml. The extension is optional: without a C++17 compiler, the build
skips it and ACES runs as before (aces.native.available() is False).
"""

import sys

from setuptools import Extension, setup

flags = ["/std:c++17", "/O2", "/EHsc"] if sys.platform == "win32" else ["-std=c++17", "-O3"]
# The C API aces.native loads with ctypes (on Windows, DLL exports must be listed).
C_API = [
    "acn_last_error",
    "acn_meds_predicates",
    "acn_result_rows",
    "acn_result_subjects",
    "acn_result_times",
    "acn_result_subject_valid",
    "acn_result_time_valid",
    "acn_result_counts",
    "acn_result_free",
]
setup(
    ext_modules=[
        Extension(
            "aces._native",
            sources=["csrc/aces_native.cc", "csrc/python_module.cc"],
            extra_compile_args=flags,
            export_symbols=[*C_API, "PyInit__native"] if sys.platform == "win32" else None,
            optional=True,
        )
    ],
)
