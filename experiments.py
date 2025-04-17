import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import casadi as ca
    import os
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from terrain_gen import generate_terrain
    from optimizer_djikstras import RailwayPathOptimizer
    return (
        LineCollection,
        LinearSegmentedColormap,
        RailwayPathOptimizer,
        ca,
        generate_terrain,
        np,
        os,
        plt,
    )


@app.cell
def _(RailwayPathOptimizer, np):
    # Create optimizer with the generated terrain
    optimizer = RailwayPathOptimizer()

    # Define start and end points (normalized coordinates)
    start = (0.2, 0.2)  # normalized
    end = (0.8, 0.8)  # normalized

    # Optimize the path with different weight configurations
    # Using fewer configurations to reduce runtime
    # (curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight)


    print(f"Optimizing path...")

    x_coords, y_coords, z_coords = optimizer.optimize_path(
        start_point=start,  # Already normalized
        end_point=end,  # Already normalized
        n_points=50
    )

    if all(v is not None for v in (x_coords, y_coords, z_coords)):
        coords = np.stack([x_coords, y_coords, z_coords], axis=-1)
        optimizer.plot_combined_profile(coords, f"Terrain Elevation and Rail Gradient Profile")
        optimizer.plot_path(coords, f"Railway Path")
    else:
        print(f"Failed to optimize path")
    return coords, end, optimizer, start, x_coords, y_coords, z_coords


if __name__ == "__main__":
    app.run()
