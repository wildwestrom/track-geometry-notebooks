# Track Planning Optimizer

A Python-based project for optimizing railway track planning through procedurally generated terrain using CasADi for numerical optimization.

## Project Overview

This project aims to find optimal railway paths through procedurally generated terrain by minimizing a combination of:
- Curvature (for passenger comfort)
- Change in curvature (for smoothness)
- Construction cost (based on terrain elevation)
- Travel time (a function of curvature, distance, and gradient)

The system uses procedural terrain generation and numerical optimization to create realistic landscapes and determine the best possible routes for railway construction.

## Core Components

### 1. Railway Path Optimizer (`casadi-experiments.py`)
The main optimization engine that:
- Takes procedurally generated terrain as input
- Defines optimization constraints:
  - Maximum allowable curvature
  - Maximum allowable gradient
  - Path must stay within terrain boundaries
- Optimizes for multiple objectives:
  - Minimize curvature (and higher derivatives for G^n continuity)
  - Minimize construction cost (terrain elevation)
  - Minimize travel time (path length)
- For debugging, uses visualization of optimized paths with:
  - Terrain elevation contours
  - Path gradient coloring
  - Detailed statistics (path length, gradients, elevations)

### 2. Terrain Generation (`terrain_gen.py`)
- Generates realistic terrain using OpenSimplex noise
- Configurable parameters for terrain characteristics:
  - Size
  - Frequency
  - Octaves
  - Persistence
  - Lacunarity
  - Smoothing
  - Seed-based generation

## Technical Details

### Optimization Approach
- Uses CasADi for symbolic optimization
- Implements a multi-objective optimization problem
- Supports different weight configurations for balancing objectives
- Includes constraints for realistic railway design:
  - Maximum gradient: 15%
  - Maximum curvature constraints
  - Terrain boundary constraints

## Dependencies

- Python 3.x
- NumPy
- CasADi
- OpenSimplex
- SciPy
- Matplotlib
- Marimo (for interactive visualization)

## Development

The project uses Nix for dependency management. To set up the development environment:

1. Ensure you have Nix installed
2. Run `direnv allow` to set up the development environment
3. The project will automatically load the correct environment when you enter the directory

The user will have a marimo notebook open to run and view code changes.

## License

GNU General Public License version 3