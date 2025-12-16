{
  description = "// fxy //";

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import inputs.systems;
      perSystem =
        { system, ... }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;

            overlays = [
              inputs.devshell.overlays.default
              inputs.hf-nix.overlays.default
              inputs.self.overlays.default
            ];

            config = {
              cudaSupport = true; # the monopoly
              cudaCapabilities = [ "12.0" ];
              rocmSupport = false; # the controlled opposition
              allowUnfree = true; # the price of admission
            };
          };
        };

      imports = [ ./default.nix ];
    };

  nixConfig = {
    extra-substituters = [
      "https://fxy-private.cachix.org"
      "https://cache.garnix.io"
      "https://cache.nixos.org"
      "https://huggingface.cachix.org"
      "https://nix-community.cachix.org"
    ];

    extra-trusted-public-keys = [
      "fxy-private.cachix.org-1:PwEBH1IvcbP+183dFzuIfDJyXeptDVDnba/StWMTkl0="
      "cache.garnix.io:CTFPyKSLcx5RMJKfLo5EEPUObbA78b0YQ2DTCJXqr9g="
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "huggingface.cachix.org-1:ynTPbLS0W8ofXd9fDjk1KvoFky9K2jhxe6r4nXAkc/o="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    hf-nix.url = "github:huggingface/hf-nix";
    nixpkgs.follows = "hf-nix/nixpkgs";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";

    agenix.url = "github:ryantm/agenix";
    agenix.inputs.nixpkgs.follows = "nixpkgs";

    agenix-shell.url = "github:aciceri/agenix-shell";
    agenix-shell.inputs.nixpkgs.follows = "nixpkgs";

    android-nixpkgs.url = "github:tadfisher/android-nixpkgs/stable";
    android-nixpkgs.inputs.nixpkgs.follows = "nixpkgs";

    bun2nix.url = "github:baileyluTCD/bun2nix?tag=1.5.1";
    bun2nix.inputs.nixpkgs.follows = "nixpkgs";
    bun2nix.inputs.systems.follows = "systems";

    devshell.url = "github:numtide/devshell";
    devshell.inputs.nixpkgs.follows = "nixpkgs";

    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";

    disko.url = "github:nix-community/disko";
    disko.inputs.nixpkgs.follows = "nixpkgs";

    mk-shell-bin.url = "github:rrbutani/nix-mk-shell-bin";

    nixos-generators.url = "github:nix-community/nixos-generators";

    nix2container.url = "github:nlewo/nix2container";
    nix2container.inputs.nixpkgs.follows = "nixpkgs";

    process-compose-flake.url = "github:Platonic-Systems/process-compose-flake";

    pwndbg.url = "github:pwndbg/pwndbg";
    pwndbg.inputs.nixpkgs.follows = "nixpkgs";

    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
}
