{
  perSystem =
    {
      inputs',
      config,
      pkgs,
      ...
    }:
    {
      python.environments.fxy = {
        workspaceRoot = ./.;
        python = pkgs.python313;
      };

      bazel.projects.fx = {
        src = ./.;
        targets = [ "//..." ];

        toolchain = {
          pythonEnv = config.packages.python-fxy;
          cuda = true;
        };

        extraNativeBuildInputs = with pkgs; [ pkg-config ];
        extraBuildInputs = with pkgs; [
          cutlass
          libmodern-cpp.pkgs
          (python3.withPackages (
            ps: with ps; [
              transformer-engine
            ]
          ))

          # TODO[b7r6]: build this with C++17...
          # opencv4
        ];

        devShellPackages = with pkgs; [
          buildifier
          inputs'.pwndbg.packages.pwndbg
          valgrind
          (python3.withPackages (
            ps: with ps; [
              transformer-engine
            ]
          ))
        ];

        devShellHook = ''
          echo "// fx // development environment"
          echo "build: bazel build //..."
          echo "test: bazel test //..."
        '';

        installPhase = ''
          mkdir -p $out/bin

          find bazel-bin/fxy -type f -executable -not -name "*.so" -not -name "*.a" | while read bin; do
            if [[ ! "$bin" =~ _test$ ]] && [[ ! "$bin" =~ MANIFEST ]]; then
              install -m 755 "$bin" "$out/bin/" 2>/dev/null || true
            fi
          done

          mkdir -p $out/lib
          find bazel-bin/fxy -name "*.so" -o -name "*.a" | while read lib; do
            if [[ ! -L "$lib" ]] || [[ ! "$(readlink "$lib")" =~ ^/nix/store ]]; then
              install -m 644 "$lib" "$out/lib/" 2>/dev/null || true
            fi
          done

          mkdir -p $out/share
          ln -s ${config.packages.fx-config}/share/fx $out/share/
        '';
      };

      packages.fx-config = pkgs.stdenvNoCC.mkDerivation {
        name = "fx-config";
        src = ./configs;

        installPhase = ''
          mkdir -p $out/share/fx
          cp *.toml $out/share/fx/
        '';

        dontBuild = true;
        dontFixup = true;
      };
    };
}
