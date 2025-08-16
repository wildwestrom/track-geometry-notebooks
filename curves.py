import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import scipy
    from scipy import special as sc
    from dataclasses import dataclass
    from typing import Callable, Dict, Tuple

    # ----------------------------
    # Core parameters and utilities
    # ----------------------------

    @dataclass(frozen=True)
    class Params:
        R: float  # design radius [m]
        L: float  # transition length [m]
        n_s: int = 1000  # samples along s

        @property
        def A(self) -> float:
            # Clothoid parameter A where R = A^2 / L
            return np.sqrt(self.R * self.L)

        @property
        def s(self) -> np.ndarray:
            return np.linspace(0.0, self.L, self.n_s)



    # Type: functions that return x(s), y(s), k(s), dk/ds(s), d²k/ds²(s)
    CurveData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    CurveFun = Callable[
        [Params],
        CurveData,
    ]

    def curve_clothoid(p: Params) -> CurveData:
        r"""
        $Θ = \frac{l^2}{2RL}$

        Where:

            - $Θ$ = The central angle at the spiral point at arc length $l$.
            - $l$ = The arc length along the curve.
        """
        s = p.s
        A = p.A
        u = s / (np.sqrt(np.pi) * A)
        S, C = sc.fresnel(u)
        x = np.sqrt(np.pi) * A * C
        y = np.sqrt(np.pi) * A * S
        k = s / (p.R * p.L)
        dkds = np.full_like(s, 1.0 / (p.R * p.L))
        d2kds2 = np.zeros_like(s)
        return x, y, k, dkds, d2kds2

    from scipy import optimize, integrate

    def _jp_X(p: Params) -> float:
        # Solve: X + X^3/(40 R^2) = L  (Civil 3D relation)
        R, L = p.R, p.L
        f = lambda X: X + X**3 / (40.0 * R**2) - L
        df = lambda X: 1.0 + (3.0 * X**2) / (40.0 * R**2)
        # Start near L; this converges quickly
        return optimize.newton(f, x0=L, fprime=df)

    def curve_cubic_parabola(p: Params) -> CurveData:
        r"""
        $$ y = \frac{x^3}{6RL} $$
        """
        s = p.s
        x = s
        y = (s ** 3) / (6.0 * p.R * p.L)
        k = s / (p.R * p.L)
        dkds = np.full_like(s, 1.0 / (p.R * p.L))
        d2kds2 = np.zeros_like(s)
        return x, y, k, dkds, d2kds2

    def curve_cubic_jp(p: Params) -> CurveData:
        r"""
        $$ y = \frac{x^3}{6RX} $$

        Where:

            - $X$ = Tangent distance at spiral-curve point from tangent-spiral point.

        $$X=L\left[\frac{10}{10+(\tan Θ_s)^2}\right]$$
        """
        s = p.s
        X = _jp_X(p)

        x = s
        y = (x**3) / (6.0 * p.R * X)

        # Curvature function
        k = x / (p.R * X)
        dkds = np.full_like(s, 1.0 / (p.R * X))
        d2kds2 = np.zeros_like(s)
        return x, y, k, dkds, d2kds2

    def curve_sine_half_wave(p: Params) -> CurveData:
        r"""
        $$y = \frac{X^2}{R}\left[\frac{a^2}{4}-\frac{1}{2 \pi^2}(1-\cos a \pi)\right]$$

        Where $a=\frac{x}{X}$ and $x$ is the distance from the start to any point on the curve and is measured along the (extended) initial tangent; $X$ is the total X at the end of the transition curve.

        Curve is only valid for $0 \leq s \leq X$.
        """
        X = _jp_X(p)
        s = np.linspace(0.0, X, p.n_s)
        a = s / X
        x = s
        y = (X ** 2 / p.R) * (0.25 * a ** 2 - (1.0 / (2.0 * np.pi ** 2)) * (1.0 - np.cos(np.pi * a)))
        # Provided k(s) and derivative (design relation)
        k = 0.5 * (1.0 / p.R) * (1.0 - np.cos(np.pi * a))
        dkds = (np.pi / (2.0 * p.R * X)) * np.sin(np.pi * a)
        d2kds2 = (np.pi**2 / (2.0 * p.R * X**2)) * np.cos(np.pi * a)
        return x, y, k, dkds, d2kds2

    def curve_bloss(p: Params) -> CurveData:
        r"""
        $$Θ=\frac{l^3}{RL^2}-\frac{l^4}{2RL^3}$$
        """
        s = p.s
        R = p.R
        L = p.L

        # Heading angle θ(s)
        theta = (s**3) / (R * L**2) - (s**4) / (2 * R * L**3)

        # Numerical integration for x(s), y(s)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x = np.concatenate(([0.0], scipy.integrate.cumulative_trapezoid(cos_theta, s)))
        y = np.concatenate(([0.0], scipy.integrate.cumulative_trapezoid(sin_theta, s)))

        # Curvature k(s) and derivative dk/ds
        k = (3 * s**2) / (R * L**2) - (2 * s**3) / (R * L**3)
        dkds = (6 * s) / (R * L**2) - (6 * s**2) / (R * L**3)
        d2kds2 = (6.0 / (R * L**2)) - (12.0 * s) / (R * L**3)
        return x, y, k, dkds, d2kds2

    def curve_sinusoidal(p: Params) -> CurveData:
        r"""
        $$\Theta(l) = \frac{l^2}{2RL} + \frac{L}{4\pi^2 R}\left[\cos\left(\frac{2\pi l}{L}\right) - 1\right]$$

        Where:
            - $\Theta(l)$ = heading angle at arc length $l$
            - $l$ = arc length along the curve
            - $R$ = design radius
            - $L$ = transition length
        """
        s = p.s
        R, L = p.R, p.L

        # Heading angle Θ(s)
        theta = (s**2) / (2.0 * R * L) + (L / (4.0 * np.pi**2 * R)) * (
            np.cos(2.0 * np.pi * s / L) - 1.0
        )

        # Coordinates by cumulative integration
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x = np.concatenate(([0.0], scipy.integrate.cumulative_trapezoid(cos_theta, s)))
        y = np.concatenate(([0.0], scipy.integrate.cumulative_trapezoid(sin_theta, s)))


        k = (s / (R * L)) - (1.0 / (2.0 * np.pi * R)) * np.sin(2.0 * np.pi * s / L)

        dkds = (1.0 / (R * L)) - (1.0 / (R * L)) * np.cos(2.0 * np.pi * s / L)
        d2kds2 = (2.0 * np.pi / (R * L**2)) * np.sin(2.0 * np.pi * s / L)
        return x, y, k, dkds, d2kds2

    # Registry for easy extension
    CURVES: Dict[str, CurveFun] = {
        "Clothoid": curve_clothoid,
        "Cubic Parabola": curve_cubic_parabola,
        "Cubic (JP)": curve_cubic_jp,
        "Sine Half-Wavelength Diminishing Tangent Curve": curve_sine_half_wave,
        "Bloss Curve": curve_bloss,
        "Sinusoidal Curve": curve_sinusoidal,
        # "Cosine": curve_cosine,
        # "Biquadratic": curve_biquadratic,
        # "Biquadratic Parabola": curve_biquadratic_parabola,
        # "Radioid": curve_radioid,
        # "WeinerBogen": curve_weiner_bogen,
    }
    return CURVES, Params


@app.cell
def _(CURVES: "Dict[str, CurveFun]", Params):
    import plotly.graph_objects as go

    def plot_curves_cartesian(p: Params, names: [str, ...]):
        fig = go.Figure()
        for name in names:
            x, y, _, _, _ = CURVES[name](p)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

        fig.update_layout(
            title="Transition Curves",
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig


    def plot_curves_1st_derivative_curvature(p: Params, names: [str, ...]):
        fig = go.Figure()
        s = p.s
        for name in names:
            _, _, _, dkds, _ = CURVES[name](p)
            fig.add_trace(go.Scatter(x=s, y=dkds, mode="lines", name=name))

        fig.update_layout(
            title=r"$\text{First derivatives of curvature}\ \frac{dk}{ds}$",
            xaxis_title="Arc length s [m]",
            yaxis_title=r"$\frac{dk}{ds} [\frac{1}{m^2}]$",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig

    def plot_curves_2nd_derivative_curvature(p: Params, names: [str, ...]):
        fig = go.Figure()
        s = p.s
        for name in names:
            _, _, _, _, d2kds2 = CURVES[name](p)
            fig.add_trace(go.Scatter(x=s, y=d2kds2, mode="lines", name=name))

        fig.update_layout(
            title=r"$\text{Second derivatives of curvature}\ \frac{d^2k}{ds^2}$",
            xaxis_title="Arc length s [m]",
            yaxis_title=r"$\frac{d^2k}{ds^2} [\frac{1}{m^3}]$",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig

    def plot_curves_3rd_derivative_curvature(p: Params, names: [str, ...]):
        fig = go.Figure()
        s = p.s
        for name in names:
            _, _, _, _, _, d3kds3 = CURVES[name](p)
            fig.add_trace(go.Scatter(x=s, y=d3kds3, mode="lines", name=name))

        fig.update_layout(
            title=r"$\text{Second derivatives of curvature}\ \frac{d^3k}{ds^3}$",
            xaxis_title="Arc length s [m]",
            yaxis_title=r"$\frac{d^3k}{ds^3} [\frac{1}{m^4}]$",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig


    def plot_error_to_clothoid(p: Params, names: [str, ...]):
        fig = go.Figure()

        if "Clothoid" not in names:
            names.append("Clothoid")

        data = {}
        for name in names:
            x, y, k, dkds, d2kds2 = CURVES[name](p)
            data[name] = (x, y, k, dkds, d2kds2)

        _, y_cl, _, _, _ = data["Clothoid"]

        for name in names:
            if name == "Clothoid":
                continue
            _, y, _, _, _ = data[name]
            err_mm = (y - y_cl) * 1000.0
            fig.add_trace(
                go.Scatter(x=p.s, y=err_mm, mode="lines", name=f"{name} Error vs Clothoid")
            )

        fig.update_layout(
            title="Lateral Error vs Clothoid",
            xaxis_title="Arc length s [m]",
            yaxis_title="Lateral error ΔY [mm]",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig

    def plot_error_to_bloss(p: Params, names: [str, ...]):
        fig = go.Figure()
        data = {}
        if "Bloss Curve" not in names:
            names = list(names) + ["Bloss Curve"]

        for name in names:
            x, y, k, dkds, d2kds2 = CURVES[name](p)
            data[name] = (x, y, k, dkds, d2kds2)

        _, y_bl, _, _, _ = data["Bloss Curve"]

        for name in names:
            if name == "Bloss Curve":
                continue
            _, y, _, _, _ = data[name]
            err_mm = (y - y_bl) * 1000.0
            fig.add_trace(
                go.Scatter(x=p.s, y=err_mm, mode="lines", name=f"{name} Error vs Bloss Curve")
            )

        fig.update_layout(
            title="Lateral Error vs Bloss Curve",
            xaxis_title="Arc length s [m]",
            yaxis_title="Lateral error ΔY [mm]",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig
    return (
        plot_curves_1st_derivative_curvature,
        plot_curves_2nd_derivative_curvature,
        plot_curves_3rd_derivative_curvature,
        plot_curves_cartesian,
        plot_error_to_bloss,
        plot_error_to_clothoid,
    )


@app.cell
def _(mo):
    R = 200.0
    L = 1500.0

    R_slider = mo.ui.slider(start=50.0, stop=4000.0, value=R)
    L_slider = mo.ui.slider(start=1.0, stop=5000.0, value=L)
    return L_slider, R_slider


@app.cell
def _(L_slider, R_slider, mo):
    mo.vstack([
        mo.hstack([R_slider, mo.md(f"R: {R_slider.value}")]), 
        mo.hstack([L_slider, mo.md(f"L: {L_slider.value}")])
    ], align="start")
    return


@app.cell
def _(
    L_slider,
    Params,
    R_slider,
    mo,
    plot_curves_1st_derivative_curvature,
    plot_curves_2nd_derivative_curvature,
    plot_curves_3rd_derivative_curvature,
    plot_curves_cartesian,
    plot_error_to_bloss,
    plot_error_to_clothoid,
):
    mo.vstack([
        mo.ui.plotly(plot_curves_cartesian(
            Params(R_slider.value, L_slider.value),
            names=["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Sine Half-Wavelength Diminishing Tangent Curve", "Sinusoidal Curve"]
        )),
        mo.ui.plotly(plot_error_to_clothoid(Params(R_slider.value, L_slider.value), names=["Cubic Parabola", "Cubic (JP)"])),
        mo.ui.plotly(plot_error_to_bloss(Params(R_slider.value, L_slider.value), names=["Sine Half-Wavelength Diminishing Tangent Curve"])),
        mo.ui.plotly(plot_curves_1st_derivative_curvature(
            Params(R_slider.value, L_slider.value),
            names=["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Sine Half-Wavelength Diminishing Tangent Curve", "Sinusoidal Curve"]
        )),
        mo.ui.plotly(plot_curves_2nd_derivative_curvature(
            Params(R_slider.value, L_slider.value),
            names=["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Sine Half-Wavelength Diminishing Tangent Curve", "Sinusoidal Curve"]
        )),
        mo.ui.plotly(plot_curves_3rd_derivative_curvature(
            Params(R_slider.value, L_slider.value),
            names=["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Sine Half-Wavelength Diminishing Tangent Curve", "Sinusoidal Curve"]
        )),
    ])
    return


if __name__ == "__main__":
    app.run()
