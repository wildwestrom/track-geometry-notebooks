{
  inputs = {
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  outputs =
    {
      self,
      nixpkgs,
      devenv,
      systems,
    }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              (
                { pkgs, lib, ... }:
                let
                  pythonPackages = pkgs.python3Packages;
                in
                {
                  # https://devenv.sh/reference/options/
                  packages = [
                    pkgs.stdenv.cc.cc.lib
                    pkgs.sqlite
                  ];

                  # https://devenv.sh/languages/
                  languages.python.enable = true;
                  languages.python.package = pythonPackages.python;
                  languages.python.uv.enable = true;
                }
              )
            ];

            # fixes libstdc++ issues and libgl.so issues
            # LD_LIBRARY_PATH = ''${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/'';
          };
        }
      );
    };
}
