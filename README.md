# Experiments in Track Geometry

I recently got very interested in track geometry for some reason, so I decided to put my experiments in some marimo notebooks.

## Core Components

### 1. Transition Curve Visualizer (`curves.py`)

Various graphs of different types of track transition curves.

### 2. Terrain Generation (`terrain_gen.py`)

Generates some terrain I may use to test a future optimizer. 

## Technical Details

### Dependencies

- Python 3.x
- NumPy
- OpenSimplex
- SciPy
- Matplotlib
- Marimo (for interactive visualization)

If I forgot something, marimo should tell you and then fetch it.

### Development

The project uses Nix for dependency management. To set up the development environment:

1. Ensure you have Nix installed
2. Run `direnv allow` to set up the development environment
3. The project will automatically load the correct environment when you enter the directory

The user will have a marimo notebook open to run and view code changes.

```
# For example...
uvx marimo edit curves.py
```

## License

MIT License