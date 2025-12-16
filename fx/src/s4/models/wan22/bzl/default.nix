{
  config.perSystem =
    { pkgs, ... }:
    {
      bazel.bzlPackage = pkgs.stdenvNoCC.mkDerivation {
        name = "bazel-bzl-files";
        src = ./.;

        installPhase = ''
          mkdir -p $out
          cp -r . $out/
        '';

        dontBuild = true;
        dontFixup = true;
      };
    };
}
