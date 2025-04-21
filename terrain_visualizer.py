import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from hashlib import sha256
    import base64
    import casadi as ca
    from scipy.interpolate import RectBivariateSpline
    from noise import snoise2
    from terrain_gen import generate_terrain
    return (
        RectBivariateSpline,
        base64,
        ca,
        gaussian_filter,
        generate_terrain,
        mo,
        np,
        plt,
        sha256,
        snoise2,
    )


@app.cell
def _(mo, np, seed_value):
    # Create UI controls
    global seed_value
    size_slider = mo.ui.slider(32, 1024, value=128, step=32, label="Terrain size")
    octaves_slider = mo.ui.slider(1, 8, value=4, step=1, label="Octaves")
    initial_frequency_slider = mo.ui.slider(
        0.0, 10, value=4.0, step=0.1, label="Initial Frequency"
    )
    persistence_slider = mo.ui.slider(
        0.01, 1.0, value=0.5, step=0.01, label="Persistence"
    )
    lacunarity_slider = mo.ui.slider(
        0.1, 10.0, value=2.7, step=0.1, label="Lacunarity"
    )
    smoothing_slider = mo.ui.slider(0, 2, value=0.0, step=0.1, label="Smoothing")

    get_seed, set_seed = mo.state(0)
    seed_button = mo.ui.button(label="Generate Seed", on_click=lambda _: _)

    rng = np.random.default_rng()
    def generate_random_seed():
        seed_value = rng.integers(0, 2**32 - 1, dtype=np.uint32)
        print(seed_value)
        set_seed(seed_value)

    seed_input = mo.ui.text(
        label="Seed",
        value=str(get_seed()),
        debounce=1000,
        on_change=lambda v: set_seed(int(v)) if v.isdigit() else 0,
    )

    # Create UI layout
    controls = mo.vstack([
        size_slider,
        initial_frequency_slider,
        octaves_slider,
        persistence_slider,
        lacunarity_slider,
        smoothing_slider,
        seed_input,
        seed_button,
    ])
    return (
        controls,
        generate_random_seed,
        get_seed,
        initial_frequency_slider,
        lacunarity_slider,
        octaves_slider,
        persistence_slider,
        rng,
        seed_button,
        seed_input,
        set_seed,
        size_slider,
        smoothing_slider,
    )


@app.cell
def _(np, plt, size_slider):
    def plot_terrain(terrain, points=[], optimal_path=None):
        size = size_slider.value

        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)

        scale = 1
        fig = plt.figure(figsize=(5*scale, 4*scale), dpi=128)

        ax = fig.add_subplot(111)
        contour = ax.contourf(X, Y, terrain, 50, cmap="terrain", vmin=0, vmax=1.0)

        if len(points) == 2:
            point_x = [p[0] for p in points]
            point_y = [p[1] for p in points]
            ax.scatter(point_x, point_y, c='red', s=30, marker='o', edgecolors='black', zorder=5)


        if optimal_path is not None:
            x_path, y_path = optimal_path
            # Normalize to [0, 1] range for plotting
            x_path_plot = np.array(x_path) / terrain.shape[1]
            y_path_plot = np.array(y_path) / terrain.shape[0]
            ax.plot(x_path_plot, y_path_plot, color='blue', linewidth=2.5, label='Optimal Path', zorder=4)

        ax.set_title('Procedurally Generated Terrain')
        fig.colorbar(contour, ax=ax)

        return plt.gca()
    return (plot_terrain,)


@app.cell
def _(
    controls,
    generate_random_seed,
    generate_terrain,
    get_seed,
    initial_frequency_slider,
    lacunarity_slider,
    mo,
    octaves_slider,
    persistence_slider,
    plot_terrain,
    seed_button,
    size_slider,
    smoothing_slider,
):
    if seed_button.value:
        generate_random_seed()
    terrain = generate_terrain(
        size_slider.value,
        initial_frequency_slider.value,
        octaves_slider.value,
        persistence_slider.value,
        lacunarity_slider.value,
        smoothing_slider.value,
        seed=int(get_seed()),
    )

    rel_start_point = (0.2, 0.2)
    rel_end_point = (0.8, 0.8)
    plot = plot_terrain(terrain, points=[rel_start_point, rel_end_point])

    # Display the UI controls and plot
    mo.hstack([controls, plot])
    return plot, rel_end_point, rel_start_point, terrain


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
