"""// `pkg-config` // repository generation"""

def _sanitize_name(name):
    """Make a valid Bazel target name."""
    return name.replace("-", "_").replace("+", "p").replace(".", "_")

def _get_package_info(ctx, pkg_name):
    """Get complete package information using `pkg-config`."""
    info = {"name": pkg_name}

    result = ctx.execute(["pkg-config", "--exists", pkg_name])
    if result.return_code != 0:
        return None

    result = ctx.execute(["pkg-config", "--cflags", "--libs", pkg_name])
    if result.return_code == 0:
        info["all_flags"] = result.stdout.strip()

    cmds = {
        "includes": ["pkg-config", "--cflags-only-I", pkg_name],
        "cflags_other": ["pkg-config", "--cflags-only-other", pkg_name],
        "lib_dirs": ["pkg-config", "--libs-only-L", pkg_name],
        "libs": ["pkg-config", "--libs-only-l", pkg_name],
        "libs_other": ["pkg-config", "--libs-only-other", pkg_name],
        "requires": ["pkg-config", "--print-requires", pkg_name],
        "requires_private": ["pkg-config", "--print-requires-private", pkg_name],
        "version": ["pkg-config", "--modversion", pkg_name],
    }

    for key, cmd in cmds.items():
        result = ctx.execute(cmd)
        if result.return_code == 0:
            info[key] = result.stdout.strip()
        else:
            info[key] = ""

    return info

def _make_include_symlink_name(inc_dir, pkg_name):
    """Create a descriptive symlink name from an include directory."""

    # `/usr/include/opencv4` -> `opencv4`
    # `/usr/local/include/torch/cuda` -> `torch_cuda`
    parts = inc_dir.rstrip("/").split("/")

    if parts[-1] == "include":
        # include dir, use package name
        return "include_%s" % _sanitize_name(pkg_name)
    elif parts[-1] == pkg_name or _sanitize_name(parts[-1]) == _sanitize_name(pkg_name):
        # already has package name
        return "include_%s" % _sanitize_name(parts[-1])
    else:
        # n.b. nested includes, e.g. `/usr/include/gtk-3.0/gtk`
        if len(parts) > 1 and "include" in parts[-2]:
            return "include_%s" % _sanitize_name(parts[-1])
        else:
            # hopefully last meaningful component...
            name_parts = []
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i]
                if part in ["usr", "local", "include", "lib", "share"]:
                    break
                name_parts.insert(0, part)

            if name_parts:
                return "include_%s" % _sanitize_name("_".join(name_parts))
            else:
                return "include_%s" % _sanitize_name(pkg_name)

def _make_cc_library(ctx, info, all_packages):
    """Generate a cc_library rule for a package."""

    pkg_name = info["name"]

    # `-I/path` ->  `/path`
    includes = []
    include_dirs = []
    for flag in [f for f in info["includes"].split(" ") if f]:
        if flag.startswith("-I") and len(flag) > 2:
            include_dirs.append(flag[2:])

    for idx, inc_dir in enumerate(include_dirs):
        link_name = _make_include_symlink_name(inc_dir, pkg_name)
        link_name = "%s_%s_%d" % (link_name, _sanitize_name(pkg_name), idx)

        ctx.symlink(inc_dir, link_name)
        includes.append(link_name)

    copts = []
    defines = []
    if info["cflags_other"]:
        for flag in [f for f in info["cflags_other"].split(" ") if f]:
            if flag.startswith("-D"):
                defines.append(flag[2:])
            else:
                copts.append(flag)

    # `linkopts`
    linkopts = []
    if info["libs"]:
        linkopts.extend([f for f in info["libs"].split(" ") if f])  # has `-l` prefix
    if info["lib_dirs"]:
        linkopts.extend([f for f in info["lib_dirs"].split(" ") if f])  # has `-L` prefix
    if info["libs_other"]:
        linkopts.extend([f for f in info["libs_other"].split(" ") if f])  # e.g. `-pthread`

    deps = []
    for req_line in info["requires"].split("\n"):
        req = req_line.strip()
        if req and req in all_packages:
            deps.append(req)

    for req_line in info["requires_private"].split("\n"):
        req = req_line.strip()
        if req and req in all_packages and req not in deps:
            deps.append(req)

    deps_str = ""
    if deps:
        deps_str = "\n    deps = [%s]," % ", ".join(['":' + _sanitize_name(d) + '"' for d in deps])

    hdrs = []
    for inc in includes:
        hdrs.extend([
            "%s/**/*.h" % inc,
            "%s/**/*.hpp" % inc,
            "%s/**/*.hh" % inc,
            "%s/**/*.hxx" % inc,
            "%s/**/*.inc" % inc,  # `abseil` does this...
            "%s/**/*.cuh" % inc,  # `CUDA`
            "%s/**/*.ipp" % inc,  # `boost` does this...
        ])

    target = _sanitize_name(pkg_name)

    version_comment = ""
    if info["version"]:
        version_comment = "# // version: %s\n" % info["version"]

    defines_attr = ""
    if defines:
        defines_attr = "\n    defines = %s," % repr(defines)

    rules = ['''{version_comment}cc_library(
    name = "{target}",
    hdrs = glob({hdrs}, allow_empty = True),
    includes = {includes},
    copts = {copts},{defines}
    linkopts = {linkopts},{deps}
    visibility = ["//visibility:public"],
)'''.format(
        version_comment = version_comment,
        target = target,
        hdrs = repr(hdrs),
        includes = repr(includes),
        copts = repr(copts),
        defines = defines_attr,
        linkopts = repr(linkopts),
        deps = deps_str,
    )]

    if target != pkg_name:
        rules.append('''alias(
    name = "{original}",
    actual = ":{target}",
    visibility = ["//visibility:public"],
)'''.format(original = pkg_name, target = target))

    return "\n\n".join(rules)

def _pkg_config_repository_impl(ctx):
    """Generate repository with all pkg-config packages."""

    # Get all available packages
    result = ctx.execute(["pkg-config", "--list-all"])
    if result.return_code != 0:
        ctx.file("BUILD.bazel", "# pkg-config not available or no packages found")
        return

    # Parse package list (name + description)
    packages = {}
    for line in result.stdout.strip().split("\n"):
        if line:
            parts = [p for p in line.split(" ") if p]  # Split on any whitespace
            if parts:
                pkg_name = parts[0]
                description = " ".join(parts[1:]) if len(parts) > 1 else ""
                packages[pkg_name] = description

    rules = []
    failed = []

    for pkg_name in sorted(packages.keys()):
        info = _get_package_info(ctx, pkg_name)
        if info:
            rule = _make_cc_library(ctx, info, packages)
            if rule:
                rules.append(rule)
        else:
            failed.append(pkg_name)

    header = ["# // auto-generated // pkg-config"]
    header.append("# // found %d packages" % len(packages))

    if failed:
        header.append("# Failed to process: %s" % ", ".join(failed))

    header.append("")

    ctx.file("BUILD.bazel", "\n".join(header) + "\n" + "\n\n".join(rules))

    manifest = []
    for name, desc in sorted(packages.items()):
        manifest.append("%s: %s" % (name, desc))
    ctx.file("packages.txt", "\n".join(manifest))

pkg_config_repository = repository_rule(
    implementation = _pkg_config_repository_impl,
    environ = ["PKG_CONFIG_PATH", "PATH"],
    local = True,
)

def _pkg_config_extension_impl(mctx):
    pkg_config_repository(name = "pkg_config")

pkg_config = module_extension(
    implementation = _pkg_config_extension_impl,
)
