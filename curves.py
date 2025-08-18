import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import scipy
    from scipy import integrate, optimize, special as sc
    from dataclasses import dataclass, field
    from typing import Callable, Dict, Tuple, Optional, Any

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

    @dataclass
    class CurveResult:
        x: np.ndarray
        y: np.ndarray
        k: np.ndarray
        derivatives: Dict[int, np.ndarray] = field(default_factory=dict)

        def get_derivative(self, order: int) -> Optional[np.ndarray]:
            return self.derivatives.get(order, None)

    CurveFun = Callable[[Params], CurveResult]
    return CurveFun, CurveResult, Dict, Params, integrate, np, optimize, sc


@app.cell(hide_code=True)
def _(CurveResult, Params, integrate, np, optimize, sc):
    def arc_len(R, L):
        def arclen(X):
            C = 1.0 / (6.0 * R * X)
            integrand = lambda x: np.sqrt(1.0 + (3.0 * C * x**2) ** 2)
            return integrate.quad(integrand, 0.0, X, limit=200)[0]

        # X must be less than L; bracket and solve
        f = lambda X: arclen(X) - L
        return optimize.brentq(f, 1e-6, L)

    def curve_clothoid(p: Params) -> CurveResult:
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
        d3kds3 = np.zeros_like(s)
        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})


    def curve_cubic_parabola(p: Params) -> CurveResult:
        r"""
        $$ y = \frac{x^3}{6RL} $$
        """
        X = arc_len(p.R, p.L)
        s = np.linspace(0.0, X, p.n_s)
        x = s
        y = (s ** 3) / (6.0 * p.R * p.L)
        k = s / (p.R * p.L)
        dkds = np.full_like(s, 1.0 / (p.R * p.L))
        d2kds2 = np.zeros_like(s)
        d3kds3 = np.zeros_like(s)
        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})

    def curve_cubic_jp(p: Params) -> CurveResult:
        """
        JP cubic defined by:
          y(x) = x^3 / (6 R X),  tan(Θ(x)) = x^2 / (2 R X)
        where X is chosen so that the arc length from 0..X equals L.
        This implementation returns x(s), y(s) sampled on s ∈ [0, L].
        """
        X = arc_len(p.R, p.L)
        s = np.linspace(0.0, X, p.n_s)
        R, L = p.R, p.L
        x = s
        y = (s**3) / (6.0 * R * X)
        k = s / (R * L)
        dkds = np.full_like(s, 1.0 / (R * L))
        d2kds2 = np.zeros_like(s)
        d3kds3 = np.zeros_like(s)
        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})

    def curve_japanese_sine(p: Params) -> CurveResult:
        r"""
        This curve goes by multiple names: Japanese Sine, Half Cosine, Sine Half-Wavelength Diminishing Tangent Curve
        $$y = \frac{X^2}{R}\left[\frac{a^2}{4}-\frac{1}{2 \pi^2}(1-\cos a \pi)\right]$$

        Where $a=\frac{x}{X}$ and $x$ is the distance from the start to any point on the curve and is measured along the (extended) initial tangent; $X$ is the total X at the end of the transition curve.

        Curve is only valid for $0 \leq s \leq X$.
        """
        X = arc_len(p.R, p.L)
        s = np.linspace(0.0, X, p.n_s)
        a = s / X
        x = s
        y = (X ** 2 / p.R) * (0.25 * a ** 2 - (1.0 / (2.0 * np.pi ** 2)) * (1.0 - np.cos(np.pi * a)))
        # Provided k(s) and derivative (design relation)
        k = 0.5 * (1.0 / p.R) * (1.0 - np.cos(np.pi * a))
        dkds = (np.pi / (2.0 * p.R * X)) * np.sin(np.pi * a)
        d2kds2 = (np.pi**2 / (2.0 * p.R * X**2)) * np.cos(np.pi * a)
        d3kds3 = -(np.pi**3 / (2.0 * p.R * X**3)) * np.sin(np.pi * a)
        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})

    def curve_bloss(p: Params) -> CurveResult:
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

        x = np.concatenate(([0.0], integrate.cumulative_trapezoid(cos_theta, s)))
        y = np.concatenate(([0.0], integrate.cumulative_trapezoid(sin_theta, s)))

        # Curvature k(s) and derivative dk/ds
        k = (3 * s**2) / (R * L**2) - (2 * s**3) / (R * L**3)
        dkds = (6 * s) / (R * L**2) - (6 * s**2) / (R * L**3)
        d2kds2 = (6.0 / (R * L**2)) - (12.0 * s) / (R * L**3)
        d3kds3 = np.full_like(s, -12.0 / (R * L**3))
        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})

    def curve_sinusoidal(p: Params) -> CurveResult:
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

        x = np.concatenate(([0.0], integrate.cumulative_trapezoid(cos_theta, s)))
        y = np.concatenate(([0.0], integrate.cumulative_trapezoid(sin_theta, s)))


        k = (s / (R * L)) - (1.0 / (2.0 * np.pi * R)) * np.sin(2.0 * np.pi * s / L)

        dkds = (1.0 / (R * L)) - (1.0 / (R * L)) * np.cos(2.0 * np.pi * s / L)
        d2kds2 = (2.0 * np.pi / (R * L**2)) * np.sin(2.0 * np.pi * s / L)
        d3kds3 = (4.0 * np.pi**2 / (p.R * p.L**3)) * np.cos(2.0 * np.pi * s / p.L)
        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})

    def curve_wiener_bogen(
        p: Params, h: float = 1.8, cant_end: float = 0.150, gauge: float = 1.435
    ) -> CurveResult:
        r"""
        The Wiener Bogen or Viennese Curve is a transition curve designed to
        optimize vehicle dynamics by linking curvature to the superelevation ramp.

        This implementation assumes a transition from a straight tangent (K=0, Cant=0)
        to a circular curve of radius R and cant `cant_end`.

        Args:
            p (Params): Standard curve parameters (s, R, L).
            h (float): Design height of the vehicle's center of gravity (meters).
                       Default is 1.8m.
            cant_end (float): Superelevation at the end of the transition (meters).
                              Default is 150mm.
            gauge (float): Track gauge (meters). Default is 1.435m for standard gauge.
        """
        s = p.s
        R, L = p.R, p.L

        # --- 1. Define the Base Function (HHMP7) and its derivatives ---
        # The base function f(u) describes the shape of the superelevation ramp,
        # where u = s/L.
        u = s / L

        # Base function f(u)
        f_u = 35 * u**4 - 84 * u**5 + 70 * u**6 - 20 * u**7

        # Derivatives of f(u) with respect to u
        df_du = 140 * u**3 - 420 * u**4 + 420 * u**5 - 140 * u**6
        d2f_du2 = 420 * u**2 - 1680 * u**3 + 2100 * u**4 - 840 * u**5
        d3f_du3 = 840 * u - 5040 * u**2 + 8400 * u**3 - 4200 * u**4
        d4f_du4 = 840 - 10080 * u + 25200 * u**2 - 16800 * u**3
        d5f_du5 = -10080 + 50400 * u - 50400 * u**2

        # Convert derivatives to be with respect to s (using chain rule d/ds = (1/L) * d/du)
        df_ds = df_du / L
        d2f_ds2 = d2f_du2 / L**2
        d3f_ds3 = d3f_du3 / L**3
        d4f_ds4 = d4f_du4 / L**4
        d5f_ds5 = d5f_du5 / L**5

        # --- 2. Calculate Curvature and its Derivatives ---
        # The final cant angle (Psi_2) is approximated by cant / gauge
        Psi_2 = cant_end / gauge

        # Curvature k(s) from the main Wiener Bogen formula
        # k(s) = (1/R)*f(s) - h*Psi_2*d2f/ds2
        k = (1 / R) * f_u - h * Psi_2 * d2f_ds2

        # Derivatives of curvature
        dkds = (1 / R) * df_ds - h * Psi_2 * d3f_ds3
        d2kds2 = (1 / R) * d2f_ds2 - h * Psi_2 * d4f_ds4
        d3kds3 = (1 / R) * d3f_ds3 - h * Psi_2 * d5f_ds5

        # --- 3. Calculate Heading Angle θ(s) by integrating k(s) ---
        # θ(s) = ∫k(s)ds = (1/R)∫f(s)ds - h*Psi_2*∫(d2f/ds2)ds
        # The integral of d2f/ds2 is simply df/ds.
        # The integral of f(s) requires integrating f(u) and multiplying by L.
        F_int_u = 7 * u**5 - 14 * u**6 + 10 * u**7 - 2.5 * u**8
        integral_f_s = L * F_int_u

        theta = (1 / R) * integral_f_s - h * Psi_2 * df_ds

        # --- 4. Calculate Coordinates by numerically integrating θ(s) ---
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x = np.concatenate(([0.0], integrate.cumulative_trapezoid(cos_theta, s)))
        y = np.concatenate(([0.0], integrate.cumulative_trapezoid(sin_theta, s)))

        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})
    return (
        curve_bloss,
        curve_clothoid,
        curve_cubic_jp,
        curve_cubic_parabola,
        curve_japanese_sine,
        curve_sinusoidal,
        curve_wiener_bogen,
    )


@app.cell
def _(CurveResult, Params, integrate, np):
    def smoothstep_curve(p: Params) -> CurveResult:
        r"""
        This spiral uses a smoothstep-based curvature function, 
        providing a $G^\infty$ continuous transition from tangent to circular arc.

        The heading angle is given by:
        $$
        \theta(l) = \frac{1}{R} \int_0^l F\!\left(\tfrac{v}{L}\right)\,dv
        $$

        where:
        - $F(z) = \dfrac{\int_0^z G(t)\,dt}{\int_0^1 G(t)\,dt}$ is the normalized smoothstep
        - $G(t) = e^{-\tfrac{1}{t(1-t)}}$
        - $l$ = arc length along the curve
        - $L$ = total length of the transition curve
        - $R$ = radius of the circular arc

        The Cartesian coordinates of the spiral are then:
        $$
        x(l) = \int_0^l \cos\!\big(\theta(v)\big)\,dv, 
        \quad 
        y(l) = \int_0^l \sin\!\big(\theta(v)\big)\,dv
        $$

        with initial conditions $x(0)=0,\ y(0)=0,\ \theta(0)=0$.

        The curvature is:
        $$\kappa(s) = \frac{1}{R} F\!\left(\frac{s}{L}\right)$$
        """
        s = p.s
        L = p.L
        R = p.R
        u = np.linspace(0.0, 1.0, p.n_s)
        def G(t):
            out = np.zeros_like(t)
            mask = (t > 0.0) & (t < 1.0)
            u_mask = t[mask]
            out[mask] = np.exp(-1.0 / (u_mask * (1.0 - u_mask)))
            return out

        G_vals = G(u)
        C = integrate.trapezoid(G_vals, u)

        F_vals = integrate.cumulative_trapezoid(G_vals, u, initial=0.0) / C

        theta_vals = (L / R) * integrate.cumulative_trapezoid(F_vals, u, initial=0.0)

        # cartesian coordinates
        cos_th = np.cos(theta_vals)
        sin_th = np.sin(theta_vals)
        x = L * integrate.cumulative_trapezoid(cos_th, u, initial=0.0)
        y = L * integrate.cumulative_trapezoid(sin_th, u, initial=0.0)

        # curvature
        k = (1.0 / R) * F_vals

        # Derivatives of G
        def h(t):
            return -1.0 / (t * (1 - t))

        def h_prime(t):
            out = np.zeros_like(t)
            # extra work to prevent division by zero
            mask = (t > 0.0) & (t < 1.0)
            num = (1 - 2 * t[mask])
            den = (t[mask] ** 2 * (1 - t[mask]) ** 2)
            out[mask] = num / den
            return out

        def h_double_prime(t):
            out = np.zeros_like(t)
            # extra work to prevent division by zero
            mask = (t > 0.0) & (t < 1.0)
            num = -2 * ((t[mask] * (1 - t[mask])) + (1 - 2 * t[mask]) ** 2)
            den = (t[mask] * (1 - t[mask])) ** 3
            out[mask] = num / den
            return out

        Gp = G_vals * h_prime(u)
        Gpp = G_vals * (h_double_prime(u) + h_prime(u) ** 2)

        # Derivatives of F
        Fp = G_vals / C
        Fpp = Gp / C
        Fppp = Gpp / C

        # Derivatives of curvature
        dkds = (1.0 / (R * L)) * Fp
        d2kds2 = (1.0 / (R * L**2)) * Fpp
        d3kds3 = (1.0 / (R * L**3)) * Fppp

        return CurveResult(x, y, k, {1: dkds, 2: d2kds2, 3: d3kds3})
    return (smoothstep_curve,)


@app.cell(hide_code=True)
def _(
    CurveFun,
    Dict,
    curve_bloss,
    curve_clothoid,
    curve_cubic_jp,
    curve_cubic_parabola,
    curve_japanese_sine,
    curve_sinusoidal,
    curve_wiener_bogen,
    smoothstep_curve,
):
    # Registry for easy extension
    CURVES: Dict[str, CurveFun] = {
        "Clothoid": curve_clothoid,
        "Cubic Parabola": curve_cubic_parabola,
        "Cubic (JP)": curve_cubic_jp,
        "Japanese Sine": curve_japanese_sine, # aka. Half Cosine, Sine Half-Wavelength Diminishing Tangent Curve
        "Bloss Curve": curve_bloss,
        "Sinusoidal Curve": curve_sinusoidal,
        # "Cosine": curve_cosine,
        # "Biquadratic": curve_biquadratic,
        # "Biquadratic Parabola": curve_biquadratic_parabola,
        # "Radioid": curve_radioid,
        "Wiener Bogen": curve_wiener_bogen,
        "Smoothstep Curve": smoothstep_curve,
    }
    return (CURVES,)


@app.cell(hide_code=True)
def _(CURVES: "Dict[str, CurveFun]", Params, jp_X):
    import plotly.graph_objects as go

    def plot_curves_cartesian(p: Params, names: [str, ...]):
        fig = go.Figure()
        for name in names:
            result = CURVES[name](p)
            fig.add_trace(go.Scatter(x=result.x, y=result.y, mode="lines", name=name))

        fig.update_layout(
            title="Transition Curves",
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig

    def plot_curvature(p: Params, names: [str, ...]):
        fig = go.Figure()
        s = p.s
        for name in names:
            curve = CURVES[name](p)
            fig.add_trace(go.Scatter(x=curve.x, y=curve.k, mode="lines", name=name))

        fig.update_layout(
            title=r"$\text{Curvature}\ \kappa(s)$",
            xaxis_title="Arc length s [m]",
            yaxis_title=r"$\kappa(s) [m]$",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig

    def ordinal_suffix(n: int) -> str:
        """Return ordinal suffix for plot titles (e.g. 1st, 2nd, 3rd, 4th)."""
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def plot_curvature_derivative(p: Params, names: [str, ...], order: int):
        fig = go.Figure()
        s = p.s
        for name in names:
            curve = CURVES[name](p)
            d = curve.get_derivative(order)
            if d is not None:
                fig.add_trace(go.Scatter(x=s, y=d, mode="lines", name=name))

        ord_str = ordinal_suffix(order)

        fig.update_layout(
            title=fr"$\text{{{ord_str} derivative of curvature}}\ \frac{{d^{order}k}}{{ds^{order}}}$",
            xaxis_title="Arc length s [m]",
            yaxis_title=fr"$\frac{{d^{order}k}}{{ds^{order}}}$",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig

    def plot_error_to_clothoid(p: Params, names: [str, ...]):
        fig = go.Figure()

        if "Clothoid" not in names:
            names = list(names) + ["Clothoid"]

        data = {name: CURVES[name](p) for name in names}
        y_cl = data["Clothoid"].y

        for name in names:
            if name == "Clothoid":
                continue
            y = data[name].y
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

    def plot_error_to_clothoid_by_x(p: Params):
        cl = CURVES["Clothoid"](p)
        xg, y_cl = cl.x, cl.y

        y_cubic = (xg**3) / (6.0 * p.R * p.L)
        X = jp_X(p)
        y_cubic_jp = (xg**3) / (6.0 * p.R * X)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xg, y=(y_cubic - y_cl) * 1000, name="Cubic − Clothoid"))
        fig.add_trace(go.Scatter(x=xg, y=(y_cubic_jp - y_cl) * 1000, name="Cubic (JP) − Clothoid"))
        fig.update_layout(
            title="Lateral Error vs Clothoid by X",
            xaxis_title="x [m]", 
            yaxis_title="Δy [mm]", 
            template="plotly_white"
        )
        return fig

    def plot_error_to_bloss(p: Params, names: [str, ...]):
        fig = go.Figure()

        if "Bloss Curve" not in names:
            names = list(names) + ["Bloss Curve"]

        data = {name: CURVES[name](p) for name in names}
        y_bl = data["Bloss Curve"].y

        for name in names:
            if name == "Bloss Curve":
                continue
            y = data[name].y
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

    def plot_error_to_my_curve(p: Params, names: [str, ...]):
        fig = go.Figure()

        if "My Curve" not in names:
            names = list(names) + ["My Curve"]

        data = {name: CURVES[name](p) for name in names}
        y_cl = data["My Curve"].y

        for name in names:
            if name == "My Curve":
                continue
            y = data[name].y
            err_mm = (y - y_cl) * 1000.0
            fig.add_trace(
                go.Scatter(x=p.s, y=err_mm, mode="lines", name=f"{name} Error vs My Curve")
            )

        fig.update_layout(
            title="Lateral Error vs My Curve",
            xaxis_title="Arc length s [m]",
            yaxis_title="Lateral error ΔY [mm]",
            legend_title="Curves",
            template="plotly_white",
        )
        return fig
    return plot_curvature, plot_curvature_derivative, plot_curves_cartesian


@app.cell(hide_code=True)
def _(mo, np):
    R = 5400.0 # Min curve radius for 350 km/h
    angle = np.radians(45)
    L = R * 2 * angle

    get_R, set_R = mo.state(R)
    get_L, set_L = mo.state(L)
    get_angle, set_angle = mo.state(angle)
    return angle, get_L, get_R, get_angle, set_L, set_R, set_angle


@app.cell(hide_code=True)
def _(angle, get_R, get_angle, mo, np, set_L, set_R, set_angle):
    max_R = 8000.0

    def on_R_change(r):
        set_R(r)
        set_L(r * 2 * get_angle())

    def on_angle_change(a):
        R = get_R()
        new_L = get_R() * 2 * a
        set_angle(a)
        set_L(new_L)


    R_slider = mo.ui.slider(start=50.0, stop=max_R, step=100, value=get_R(), on_change=on_R_change)
    angle_slider = mo.ui.slider(start=0.0, stop=np.radians(360), step=np.radians(1), value=angle, on_change=on_angle_change)
    #L_slider = mo.ui.slider(start=10.0, stop=max_R * 2 * np.radians(360.0), step=100, value=get_L(), on_change=set_L)
    return R_slider, angle_slider


@app.cell(hide_code=True)
def _(R_slider, angle_slider, get_L, mo, np):
    mo.vstack([
        mo.hstack([R_slider, f"{R_slider.value:0.0f} m"]),
        mo.hstack([angle_slider, f"{np.degrees(angle_slider.value):0.1f}°"]),
        mo.hstack([f"Transition Length: {get_L():0.1f} m"])
    ], align="start")
    return


@app.cell(hide_code=True)
def _(
    Params,
    get_L,
    get_R,
    mo,
    plot_curvature,
    plot_curvature_derivative,
    plot_curves_cartesian,
):
    mo.vstack([
        mo.ui.plotly(plot_curves_cartesian(Params(get_R(), get_L()), names=["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Japanese Sine", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve",])),
        #mo.ui.plotly(plot_curves_cartesian(Params(get_R(), get_L()), names=["Clothoid", "Bloss Curve", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve"])),
        mo.ui.plotly(plot_curvature(Params(get_R(), get_L()), names=["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Japanese Sine", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve",])),
        #mo.ui.plotly(plot_curvature(Params(get_R(), get_L()), names=["Japanese Sine", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve",])),
        #mo.ui.plotly(plot_curves_cartesian(Params(get_R(), get_L()), names=["Smoothstep Curve"])),
        #mo.ui.plotly(plot_error_to_my_curve(Params(get_R(), get_L()), names=["Clothoid", "Bloss Curve", "Sinusoidal Curve", "Smoothstep Curve"])),
        #mo.ui.plotly(plot_error_to_clothoid(Params(get_R(), get_L()), names=["Cubic Parabola", "Cubic (JP)", "Smoothstep Curve"])),
        #mo.ui.plotly(plot_error_to_clothoid_by_x(Params(get_R(), get_L()))),
        #mo.ui.plotly(plot_error_to_bloss(Params(get_R(), get_L()), names=["Japanese Sine", "Smoothstep Curve"])),
        mo.ui.plotly(plot_curvature_derivative(Params(get_R(), get_L()), ["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Japanese Sine", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve"], 1)),
        mo.ui.plotly(plot_curvature_derivative(Params(get_R(), get_L()), ["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Japanese Sine", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve"], 2)),
        mo.ui.plotly(plot_curvature_derivative(Params(get_R(), get_L()), ["Clothoid", "Cubic Parabola", "Cubic (JP)", "Bloss Curve", "Japanese Sine", "Sinusoidal Curve", "Wiener Bogen", "Smoothstep Curve"], 3))
    ])
    return


if __name__ == "__main__":
    app.run()
