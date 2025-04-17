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
    from optimizer import RailwayPathOptimizer
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
    weight_configurations = [
        (1.0, 1.0, 1.0, 0.0, 1.0, "No terrain cost"),
        # (1.0, 1.0, 1.0, 0.2, 1.5, "Terrain aware"),  # More emphasis on path length to avoid excessive detours
        # (2.0, 2.0, 1.0, 1.0, 0.5, "Low Curvature"),  # Prioritize low curvature
        # (1.0, 1.0, 1.0, 5.0, 0.5, "Low Cost"),  # Prioritize low cost
    ]

    for weights in weight_configurations:
        curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight, label = weights

        print(f"Optimizing path with {label} configuration...")

        # For no terrain cost case, use straight line initialization without perturbation
        if terrain_cost_weight == 0:
            n_points = 20  # Fewer points for straight line case
        else:
            n_points = 40

        x_coords, y_coords, z_coords = optimizer.optimize_path(
            start_point=start,  # Already normalized
            end_point=end,  # Already normalized
            max_curvature=0.3,
            max_gradient=0.15,
            weights=(curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight),
            n_points=n_points
        )

        if all(v is not None for v in (x_coords, y_coords, z_coords)):
            coords = np.stack([x_coords, y_coords, z_coords], axis=-1)
            optimizer.plot_path(coords, terrain, f"Railway Path: {label} configuration")
            optimizer.plot_combined_profile(coords, terrain, f"Terrain Elevation and Rail Gradient Profile: {label}")
        else:
            print(f"Failed to optimize path with {label} configuration")
    return (
        coords,
        curvature_change_weight,
        curvature_weight,
        end,
        gradient_weight,
        label,
        n_points,
        optimizer,
        start,
        terrain,
        terrain_cost,
        terrain_cost_weight,
        terrain_size,
        time_weight,
        weight_configurations,
        weights,
        x_coords,
        y_coords,
        z_coords,
    )


if __name__ == "__main__":
    app.run()
