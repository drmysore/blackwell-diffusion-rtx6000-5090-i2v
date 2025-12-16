# // fxy // python // environments

This document describes how to use Python environment management for standard `uv` projects using
`pyproject.toml` with flake-parts.

## // overview

The Python environment management works seamlessly with `uv` projects. It expects a standard
`pyproject.toml` file with dependencies managed by `uv`.

## // basic // usage

### // project // structure

A typical `uv` project structure:

```
my-project/
├── pyproject.toml
├── uv.lock
├── src/
│   └── my_app/
│       ├── __init__.py
│       └── main.py
└── flake.nix
```

Example `pyproject.toml`:

```toml
[project]
name = "my-app"
version = "0.1.0"
description = "A FastAPI application"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1.0",
]
```

### // quick // start

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    fxy.url = "github:your/fxy";
    uv2nix.url = "github:your/uv2nix";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      
      perSystem = { pkgs, ... }:
      let
        workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
          workspaceRoot = ./.; 
        };
        
        pyEnv = inputs.fxy.lib.python.mkPython {
          inherit pkgs workspace;
        };
        
        pythonEnv = pyEnv.mkVirtualEnv "my-app" workspace.deps.all;
      in
      {
        devShells.default = pyEnv.mkShell { env = pythonEnv; };
        
        packages.default = pyEnv.mkApp {
          name = "my-app";
          script = ./src/my_app/main.py;
          env = pythonEnv;
        };
      };
    };
}
```

## // `uv` // projects

### // development // environments

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
    };
    
    # Include all dependencies
    pythonEnv = pyEnv.mkVirtualEnv "dev" workspace.deps.all;
  in
  {
    devShells = {
      default = pyEnv.mkShell { 
        env = pythonEnv;
        packages = [ pkgs.postgresql ];
      };
      
      # Minimal shell with just runtime deps
      prod = pyEnv.mkShell { 
        env = pyEnv.mkVirtualEnv "prod" workspace.deps.default;
      };
    };
  };
}
```

### // applications

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
    };
    
    pythonEnv = pyEnv.mkVirtualEnv "my-service" workspace.deps.default;
  in
  {
    packages = {
      my-service = pyEnv.mkApp {
        name = "my-service";
        script = ./src/my_service/main.py;
        env = pythonEnv;
      };
      
      # Alternative: as a module
      my-cli = pyEnv.mkApp {
        name = "my-cli";
        script = "-m my_cli.main";  # Runs as python -m
        env = pythonEnv;
      };
    };
  };
}
```

### // custom // derivations

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
    };
    
    pythonEnv = pyEnv.mkVirtualEnv "web-app" workspace.deps.default;
  in
  {
    packages.web-app = pkgs.stdenv.mkDerivation {
      pname = "web-app";
      version = workspace.package.version;
      
      src = ./.;
      
      buildInputs = [ pythonEnv ];
      
      installPhase = ''
        mkdir -p $out/{bin,share/web-app}
        
        # Copy static assets
        cp -r static $out/share/web-app/
        
        # Create runner
        cat > $out/bin/web-app << EOF
        #!${pkgs.bash}/bin/bash
        export STATIC_ROOT=$out/share/web-app/static
        exec ${pythonEnv}/bin/python -m uvicorn my_app.main:app \
          --host 0.0.0.0 \
          --port \''${PORT:-8000} \
          "\$@"
        EOF
        chmod +x $out/bin/web-app
      '';
    };
  };
}
```

## // nixos // module

```nix
# flake.nix
{ inputs, ... }:
{
  flake = {
    nixosModules.my-service = import ./modules/my-service.nix inputs;
  };
  
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
    };
    
    pythonEnv = pyEnv.mkVirtualEnv "my-service" workspace.deps.default;
  in
  {
    packages.my-service = pyEnv.mkApp {
      name = "my-service";
      script = ./src/my_service/main.py;
      env = pythonEnv;
    };
  };
}
```

```nix
# modules/my-service.nix
inputs:
{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.my-service;
  
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
    workspaceRoot = ./..;
  };
  
  pyEnv = inputs.fxy.lib.python.mkPython {
    inherit pkgs workspace;
  };
  
  pythonEnv = pyEnv.mkVirtualEnv "my-service" workspace.deps.default;
  
  app = pyEnv.mkApp {
    name = "my-service"; 
    script = ../src/my_service/main.py;
    env = pythonEnv;
  };
in
{
  options.services.my-service = {
    enable = mkEnableOption "My Service";
    
    port = mkOption {
      type = types.port;
      default = 8000;
    };
    
    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
    };
  };
  
  config = mkIf cfg.enable {
    systemd.services.my-service = {
      description = "My Service";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];
      
      environment = cfg.environment // {
        PORT = toString cfg.port;
      };
      
      serviceConfig = {
        ExecStart = "${app}/bin/my-service";
        Restart = "always";
        DynamicUser = true;
      };
    };
  };
}
```

## // advanced // configuration

### // system // libraries

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
      extraSystemLibs = [ 
        pkgs.postgresql.lib
        pkgs.libxml2
      ];
    };
  in
  {
    # ...
  };
}
```

### // package // overrides

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
      extraOverrides = final: prev: {
        # Override specific packages
        pillow = prev.pillow.overridePythonAttrs (old: {
          nativeBuildInputs = old.nativeBuildInputs ++ [ pkgs.libjpeg ];
        });
      };
    };
  in
  {
    # ...
  };
}
```

### // python // version

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    mkEnvForPython = python: inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace python;
    };
    
    py311Env = mkEnvForPython pkgs.python311;
    py312Env = mkEnvForPython pkgs.python312;
  in
  {
    packages = {
      app-py311 = py311Env.mkApp {
        name = "app";
        script = ./src/app.py;
        env = py311Env.mkVirtualEnv "app" workspace.deps.default;
      };
      
      app-py312 = py312Env.mkApp {
        name = "app";
        script = ./src/app.py;
        env = py312Env.mkVirtualEnv "app" workspace.deps.default;
      };
    };
  };
}
```

## // common // patterns

### // workspace // dependencies

`uv2nix` provides different dependency sets from your workspace:

```nix
let
  workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
    workspaceRoot = ./.; 
  };
  
  pyEnv = inputs.fxy.lib.python.mkPython {
    inherit pkgs workspace;
  };
in
{
  # All dependencies (including dev)
  devEnv = pyEnv.mkVirtualEnv "dev" workspace.deps.all;
  
  # Only runtime dependencies
  prodEnv = pyEnv.mkVirtualEnv "prod" workspace.deps.default;
  
  # If you have groups defined in pyproject.toml
  testEnv = pyEnv.mkVirtualEnv "test" workspace.deps.groups.test;
}
```

### // multi // output

```nix
{ inputs, ... }:
{
  perSystem = { pkgs, ... }:
  let
    workspace = inputs.uv2nix.lib.workspace.loadWorkspace { 
      workspaceRoot = ./.; 
    };
    
    pyEnv = inputs.fxy.lib.python.mkPython {
      inherit pkgs workspace;
    };
    
    pythonEnv = pyEnv.mkVirtualEnv "my-app" workspace.deps.default;
  in
  {
    packages.my-app = pkgs.stdenv.mkDerivation {
      pname = "my-app";
      version = workspace.package.version;
      
      outputs = [ "out" "doc" ];
      
      src = ./.;
      
      buildInputs = [ pythonEnv pkgs.sphinx ];
      
      buildPhase = ''
        # Build documentation
        sphinx-build docs $doc
      '';
      
      installPhase = ''
        mkdir -p $out/bin
        
        cat > $out/bin/my-app << EOF
        #!${pkgs.bash}/bin/bash
        exec ${pythonEnv}/bin/python -m my_app "\$@"
        EOF
        chmod +x $out/bin/my-app
      '';
    };
  };
}
```

## Tips

1. **Always use workspace**: Load your workspace with `inputs.uv2nix.lib.workspace.loadWorkspace` to
   ensure proper dependency resolution.

2. **Dependency sets**: Use `workspace.deps.all` for development and `workspace.deps.default` for
   production.

3. **Module paths**: Use `-m module.path` syntax in scripts when your package is installed in
   editable mode.

4. **Version management**: Access package version via `workspace.package.version`.

5. **Source preference**: The default `sourcePreference = "wheel"` is usually optimal, but you can
   change it if needed.
