# Track Planning Optimizer

A Python-based project for optimizing railway track planning through procedurally generated terrain using CasADi for numerical optimization.

## Project Overview

I'd like to create a limited scope prototype that allows for creating MVC splines like in spiro.c/spiro.h but without `libspiro`'s limitations. The major limitation of `libspiro` is that it doesn't truly solve the MVC functional, and it only generates planar curves. The calculations must be truly 3D as opposed to 2.5D, so that torsion, not just curvature, can be taken into account.

The specific function I'd like to optimize for is

$$ E_{SpaceMVC}[\kappa(s),\tau(s)] = \int_0^l (\kappa')^2 + (\tau')^2 ds $$

One application I'd like to use this for is high speed transportation. In particular, the tracks for superconducting maglev trains. These vehicles can reach speeds of up to 600km/h so if the track should have curves, then the curves must be VERY smooth (G4 continuity, minimizing curvature and torsion).

The workflow for this type of program should be something like this: 
- Set a series of control points
- See an approximation of the route
- Edit/add control points to match constraints of the project
- See the route approximation again

The visualizations (top-down alignment, gradient, elevation, curvature, forces, speed limits, etc.) should guide the user as to what choices to make, but the initial path should be optimal as the MVC is already a sort of optimization.

Control points example: [end]-[straight-to-curve]-[curve-point]-[curve-to-straight]-[straight-to-curve]-[curve-point]-[curve-point]-[curve-to-straight]-[end]

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