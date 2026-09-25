# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.nixpkgs.follows = "nixpkgs";
    jupyter.url = "github:kirelagin/jupyter.nix";
    jupyter.inputs.nixpkgs.follows = "nixpkgs";
    jupyter.inputs.flake-utils.follows = "flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, jupyter }:
    flake-utils.lib.eachDefaultSystem (system: {
      packages = {
        jupyter = jupyter.lib.makeJupyterLab {
          pkgs = nixpkgs.legacyPackages.${system};
          kernels = {
            "python3".ipykernel = {
              packages = pp: with pp; [
                numpy
                polars
                matplotlib
                scipy
                wget
              ];
              withPlotly = true;
            };
          };
        };
      };
    });
}
