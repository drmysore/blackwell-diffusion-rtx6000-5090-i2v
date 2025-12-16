{ inputs, ... }:
{
  imports = [
    inputs.devshell.flakeModule
    inputs.git-hooks.flakeModule
    inputs.treefmt-nix.flakeModule

    ./nix/flake-modules

    # TODO[b7r6]: this is pretty clearly going up ^ there,
    # i'm ready to bite the bullet on `mkPerSystemOption`...
    ./android
    ./api/fal.ai
    ./fx
    ./idoru/apps/react-ono-sendai
    ./idoru/inference-server
    ./idoru/server
    ./nix/configurations
    ./nix/nixos
    ./nix/overlays
    ./nix/packages
    ./secrets
  ];

  flake.lib = {
    # TODO[b7r6]: you're getting clean up too buddy...
    agenix-shell-auto = import ./nix/flake-modules/agenix-shell-auto.nix { inherit inputs; };
  };

  perSystem =
    { pkgs, ... }:
    {
      devShells.default = pkgs.mkShellNoCC {
        name = "fxy";

        packages = with pkgs; [
          biome
          bun
          git
          nodejs_24
          rclone
          uv
        ];

        shellHook = ''
          echo "// fxy // src"
        '';
      };
    };
}
