load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("//toolchain:variables.bzl", "PYTHON_PATHS")

def python_torch_library(name, **kwargs):
    """A cc_library that provides Python/PyTorch embedding support."""
    cc_library(
        name = name,
        deps = [
            "@pkg_config//:libtorch",
            "@pkg_config//:pybind11",
            "@pkg_config//:python3",
        ],
        defines = [
            "PYTHON_SITE_PACKAGES='\"" + PYTHON_PATHS["site_packages"] + "\"'",
            "TORCH_LIB_PATH='\"" + PYTHON_PATHS["torch_lib"] + "\"'",
        ],
        linkopts = [
            "-L" + PYTHON_PATHS["python_lib"],
            "-L" + PYTHON_PATHS["torch_lib"],
            "-lpython3.13",
            "-ltorch_python",
            "-Wl,-rpath," + PYTHON_PATHS["python_lib"],
            "-Wl,-rpath," + PYTHON_PATHS["torch_lib"],
        ],
        **kwargs
    )

def python_torch_binary(name, srcs, deps = [], **kwargs):
    """A cc_binary with Python/PyTorch embedding support."""
    cc_binary(
        name = name,
        srcs = srcs,
        deps = deps + ["//fxy/util:python_torch_embed"],
        **kwargs
    )
