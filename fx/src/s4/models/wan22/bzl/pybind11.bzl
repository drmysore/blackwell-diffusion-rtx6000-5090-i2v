load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_python//python:defs.bzl", "py_library")
load("//toolchain:variables.bzl", "TOOLCHAIN_VARS")

def pybind11_extension(name, srcs, deps = [], **kwargs):
    """Creates a pybind11 Python extension module."""

    cc_binary(
        name = name + ".so",
        srcs = srcs,
        deps = deps + [
            "@pkg_config//:python",
            "@pkg_config//:pybind11",
        ],
        linkshared = True,
        linkstatic = False,
        linkopts = TOOLCHAIN_VARS["link_flags"],
        **kwargs
    )

    py_library(
        name = name,
        data = [":" + name + ".so"],
        imports = ["."],
        visibility = ["//visibility:public"],
    )
