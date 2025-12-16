{
  perSystem =
    { pkgs, ... }:
    {
      python.environments.fx = {
        workspaceRoot = ./.;
        python = pkgs.python313;
        cudaSupport = true;
      };
    };
}
