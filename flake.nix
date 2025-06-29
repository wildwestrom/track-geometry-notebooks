{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
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
                  pythonPackages = pkgs.python313Packages;
                in
                {
                  # https://devenv.sh/reference/options/
                  packages = with pkgs; [
                    stdenv.cc.cc.lib
                    watchdog
                    libz
                    pyright
                  ];

                  enterShell = ''
                    export USE_EXTERNAL_RAYLIB=1
                  '';

                  languages.python.enable = true;
                  languages.python.package = pkgs.python313;
                  languages.python.venv.enable = true;
                }
              )
            ];
          };
        }
      );
    };
}
