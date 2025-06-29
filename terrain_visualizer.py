import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from terrain_gen import TerrainGen

    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    return TerrainGen, mo, np, px


@app.cell
def _(mo):
    get_seed, set_seed = mo.state(0)
    return get_seed, set_seed


@app.cell
def _(np, set_seed):
    def generate_random_seed():
        rng = np.random.default_rng()
        seed_value = rng.integers(0, 2**32 - 1, dtype=np.uint32)
        set_seed(seed_value)
    return (generate_random_seed,)


@app.cell
def _(generate_random_seed, mo):
    seed_button = mo.ui.button(
        label="Generate Seed", on_click=lambda _: generate_random_seed()
    )
    return (seed_button,)


@app.cell
def _(get_seed, mo):
    seed_form = mo.ui.text(
        label="Seed",
        value=str(get_seed()),
    ).form(submit_button_label="Update Seed")
    return (seed_form,)


@app.cell
def _(mo, seed_button, seed_form):
    size_slider = mo.ui.slider(8, 1024, value=32, step=8, label="Terrain size")
    octaves_slider = mo.ui.slider(1, 8, value=4, step=1, label="Octaves")
    initial_frequency_slider = mo.ui.slider(
        0.0, 10, value=4.0, step=0.1, label="Initial Frequency"
    )
    persistence_slider = mo.ui.slider(
        0.01, 1.0, value=0.5, step=0.01, label="Persistence"
    )
    lacunarity_slider = mo.ui.slider(0.1, 10.0, value=2.7, step=0.1, label="Lacunarity")
    smoothing_slider = mo.ui.slider(0, 2, value=0.0, step=0.1, label="Smoothing")
    baseline_slider = mo.ui.slider(0.0, 1.0, value=0.2, step=0.01, label="Baseline")
    dip_depth_slider = mo.ui.slider(0.0, 0.5, value=0.1, step=0.01, label="Dip Depth")
    peak_sharpness_slider = mo.ui.slider(
        0.5, 5.0, value=1.0, step=0.1, label="Peak Sharpness"
    )
    epsilon_slider = mo.ui.slider(0.0001, .01, value=0.0001, step=0.0001, label="Epsilon (length vs slope)")
    max_segments_input = mo.ui.number(value=5, start=1, stop=50, step=1, label="Max segments")

    controls = mo.vstack(
        [
            size_slider,
            initial_frequency_slider,
            octaves_slider,
            persistence_slider,
            lacunarity_slider,
            smoothing_slider,
            baseline_slider,
            dip_depth_slider,
            peak_sharpness_slider,
            epsilon_slider,
            max_segments_input,
            seed_form.left(),
            seed_button.left(),
        ]
    )
    return (
        baseline_slider,
        controls,
        dip_depth_slider,
        epsilon_slider,
        initial_frequency_slider,
        lacunarity_slider,
        max_segments_input,
        octaves_slider,
        peak_sharpness_slider,
        persistence_slider,
        size_slider,
        smoothing_slider,
    )


@app.function
def matplotlib_to_plotly_cmap(cmap_name="terrain", n_colors=256):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    colorscale = []
    for i in range(n_colors):
        frac = i / (n_colors - 1)
        r, g, b, a = cmap(frac)
        colorscale.append([frac, f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"])
    return colorscale


@app.cell
def _(np, px, size_slider):
    def plot_terrain_interactive(terrain, points=[], optimal_paths=None, H=None):
        """Interactive 3D terrain visualization using Plotly"""

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        size = size_slider.value
        colorscale = matplotlib_to_plotly_cmap("terrain")

        # Create subplots: 2D contour and 3D surface
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "contour"}, {"type": "surface"}]],
            subplot_titles=("2D Terrain Contour", "3D Interactive Terrain")
        )

        # 2D contour plot
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        fig.add_trace(
            go.Contour(
                x=x, y=y, z=terrain,
                colorscale=colorscale,
                showscale=True,
                name="Terrain"
            ),
            row=1, col=1
        )

        # 3D surface plot
        X, Y = np.meshgrid(x, y)
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=terrain,
                colorscale=colorscale,
                showscale=False,
                name="Terrain Surface"
            ),
            row=1, col=2
        )

        # Add start and end points
        if len(points) == 2:
            point_x = [p[0] for p in points]
            point_y = [p[1] for p in points]
            # Add points to 2D plot
            fig.add_trace(
                go.Scatter(
                    x=point_x, y=point_y,
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='circle'),
                    name="Start/End Points",
                    showlegend=False
                ),
                row=1, col=1
            )
            # Add points to 3D plot
            if H is not None:
                point_z = [H(p[0], p[1])[0, 0] for p in points]
                fig.add_trace(
                    go.Scatter3d(
                        x=point_x, y=point_y, z=point_z,
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='circle'),
                        name="Start/End Points",
                        showlegend=False
                    ),
                    row=1, col=2
                )

        # Add optimal paths
        if optimal_paths is not None and H is not None:
            colors = px.colors.sequential.Rainbow
            for i, optimal_path in enumerate(optimal_paths):
                if len(optimal_path[0]) > 0:
                    x_path, y_path = optimal_path
                    color = colors[i % len(colors)]
                    # Add path to 2D plot
                    fig.add_trace(
                        go.Scatter(
                            x=x_path, y=y_path,
                            mode='lines',
                            line=dict(color=color, width=3),
                            name=f"Path {i+1}",
                            showlegend=True
                        ),
                        row=1, col=1
                    )
                    # Add path to 3D plot with height sampling
                    num_path_points = min(100, len(x_path))
                    path_heights = []
                    path_x_sampled = []
                    path_y_sampled = []
                    for j in range(num_path_points):
                        t = j / (num_path_points - 1)
                        idx = int(t * (len(x_path) - 1))
                        if idx >= len(x_path) - 1:
                            x_p, y_p = x_path[-1], y_path[-1]
                        else:
                            t_local = t * (len(x_path) - 1) - idx
                            x_p = x_path[idx] * (1 - t_local) + x_path[idx + 1] * t_local
                            y_p = y_path[idx] * (1 - t_local) + y_path[idx + 1] * t_local
                        height = H(x_p, y_p)[0, 0]
                        path_heights.append(height)
                        path_x_sampled.append(x_p)
                        path_y_sampled.append(y_p)
                    fig.add_trace(
                        go.Scatter3d(
                            x=path_x_sampled, y=path_y_sampled, z=path_heights,
                            mode='lines',
                            line=dict(color=color, width=5),
                            name=f"Path {i+1}",
                            showlegend=False
                        ),
                        row=1, col=2
                    )

        # Update layout
        fig.update_layout(
            title="Interactive Terrain Visualization",
            width=1000,
            height=600,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Height",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            scene2=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Height"
            ),
            legend=dict(
                x=-0.15,
                y=1,
                xanchor='left',
                yanchor='top',
                orientation='v',
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='black',
                borderwidth=1,
            )
        )
        return fig
    return (plot_terrain_interactive,)


@app.cell
def _(np):
    from scipy.special import fresnel

    def compute_clothoid_path(R, L_cl, num_points=100):
        """
        Computes the local coordinates of a clothoid path.
        The path starts at (0,0) with a horizontal tangent.

        Args:
            R (float): Radius of the circular arc it connects to.
            L_cl (float): Length of the clothoid curve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - s (np.ndarray): The arc length array from 0 to L_cl.
                - path (np.ndarray): A (num_points, 2) array of (x, y) coordinates.
        """
        if R == 0 or L_cl == 0:
            return np.array([[0, 0]])

        # The clothoid parameter A^2 = R * L
        A = np.sqrt(R * L_cl)
        s = np.linspace(0, L_cl, num_points)

        # Fresnel integrals S(t) and C(t)
        scaling_factor = A * np.sqrt(np.pi)
        S, C = fresnel(s / scaling_factor)

        # Scale by the clothoid parameter
        x_path = scaling_factor * C
        y_path = scaling_factor * S

        return s, np.column_stack([x_path, y_path])
    return


@app.cell
def _(
    TerrainGen,
    baseline_slider,
    dip_depth_slider,
    get_seed,
    initial_frequency_slider,
    lacunarity_slider,
    octaves_slider,
    peak_sharpness_slider,
    persistence_slider,
    size_slider,
    smoothing_slider,
):
    gen = TerrainGen()
    # Override default values from controls
    gen.size = size_slider.value
    gen.frequency = initial_frequency_slider.value
    gen.octaves = octaves_slider.value
    gen.persistence = persistence_slider.value
    gen.lacunarity = lacunarity_slider.value
    gen.smoothing = smoothing_slider.value
    # Set new terrain shaping parameters
    gen.baseline = baseline_slider.value
    gen.dip_depth = dip_depth_slider.value
    gen.peak_sharpness = peak_sharpness_slider.value

    terrain = gen.generate_terrain(int(get_seed()))
    return (terrain,)


@app.cell
def _(np, size_slider, terrain):
    from scipy.interpolate import RectBivariateSpline

    size = size_slider.value
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)

    # bicubic interpolation of the generated terrain so we can get a continuous function representing the height of our terrain
    H = RectBivariateSpline(x, y, terrain)
    return (H,)


@app.cell
def _(epsilon_slider, max_segments_input, np):
    from scipy.optimize import minimize

    start_point = (0.2, 0.2)
    end_point = (0.8, 0.8)

    epsilon = epsilon_slider.value
    segments = max_segments_input.value

    def calculate_path_geometry(x_N, start_point, end_point):
        N = len(x_N) // 4
        vertices = [(x_N[4 * i], x_N[4 * i + 1]) for i in range(N)]
        return {
            "total_length": sum(
                np.sqrt(
                    (vertices[i][0] - (vertices[i - 1] if i > 0 else start_point)[0]) ** 2
                    + (vertices[i][1] - (vertices[i - 1] if i > 0 else start_point)[1]) ** 2
                )
                for i in range(N)
            )
            + np.sqrt(
                (end_point[0] - vertices[-1][0]) ** 2
                + (end_point[1] - vertices[-1][1]) ** 2
            ),
            "vertices": vertices,
            "start_point": start_point,
            "end_point": end_point,
        }

    def get_point_on_path(s, path_geometry):
        vertices = path_geometry["vertices"]
        start_point = path_geometry["start_point"]
        end_point = path_geometry["end_point"]
        path_points = [start_point] + vertices + [end_point]
        cumulative_distances = [0]
        for i in range(1, len(path_points)):
            dist = np.sqrt(
                (path_points[i][0] - path_points[i - 1][0]) ** 2
                + (path_points[i][1] - path_points[i - 1][1]) ** 2
            )
            cumulative_distances.append(cumulative_distances[-1] + dist)
        for i in range(len(cumulative_distances) - 1):
            if cumulative_distances[i] <= s < cumulative_distances[i + 1]:
                t = (s - cumulative_distances[i]) / (
                    cumulative_distances[i + 1] - cumulative_distances[i]
                )
                p1, p2 = path_points[i], path_points[i + 1]
                return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
        return end_point

    def get_path_tangent(s, path_geometry, ds=0.01):
        s_plus = min(s + ds, path_geometry["total_length"])
        s_minus = max(s - ds, 0)
        point_plus = get_point_on_path(s_plus, path_geometry)
        point_minus = get_point_on_path(s_minus, path_geometry)
        dx = point_plus[0] - point_minus[0]
        dy = point_plus[1] - point_minus[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            return (dx / length, dy / length)
        else:
            return (0, 0)

    def objective_function_J(
        x_N,
        start_point,
        end_point,
        H,
        epsilon,
    ):
        SLOPE_PENALTY_SCALE = 1000  # Strongly penalize slope
        path_geom = calculate_path_geometry(x_N, start_point, end_point)
        total_length = path_geom["total_length"]
        num_steps = 100
        ds = total_length / num_steps
        total_cost = 0.0
        for i in range(num_steps):
            s = i * ds
            point_xy = get_point_on_path(s, path_geom)
            tangent = get_path_tangent(s, path_geom)
            dx = 0.01
            dy = 0.01
            dH_dx = (H(point_xy[0] + dx, point_xy[1])[0, 0] - H(point_xy[0] - dx, point_xy[1])[0, 0]) / (2 * dx)
            dH_dy = (H(point_xy[0], point_xy[1] + dy)[0, 0] - H(point_xy[0], point_xy[1] - dy)[0, 0]) / (2 * dy)
            grad = np.array([dH_dx, dH_dy])
            slope_along_path = np.dot(grad, tangent)
            integrand = epsilon + (1 - epsilon) * SLOPE_PENALTY_SCALE * (slope_along_path ** 2)
            total_cost += integrand * ds
        return total_cost
    return end_point, epsilon, segments, start_point


@app.cell
def _(
    H,
    end_point,
    epsilon,
    plot_terrain_interactive,
    segments,
    start_point,
    terrain,
):
    from optimizer import optimize_paths

    optimal_paths, results = optimize_paths(H, start_point, end_point, epsilon, segments)
    interactive_plot = plot_terrain_interactive(
        terrain, points=[start_point, end_point], optimal_paths=optimal_paths, H=H
    )
    return (interactive_plot,)


@app.cell
def _(controls, generate_random_seed, seed_button, seed_form, set_seed):
    if seed_button.value:
        generate_random_seed()
    if seed_button.value and seed_form.value:
        set_seed(int(seed_form.value))

    # Display the UI controls and plot
    controls
    return


@app.cell
def _(interactive_plot, mo):
    mo.md("## Interactive 3D Visualization (Plotly)")
    interactive_plot
    return


if __name__ == "__main__":
    app.run()
