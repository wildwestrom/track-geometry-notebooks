import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from scipy.integrate import trapezoid, cumulative_trapezoid
    import matplotlib, matplotlib.pyplot as plt

    R = 2000.0
    L = R * np.radians(45)
    N = 4000
    u = np.linspace(0.0, 1.0, N)
    s_grid = L * u

    def G(t):
        out = np.zeros_like(t)
        mask = (t > 0.0) & (t < 1.0)
        u = t[mask]
        out[mask] = np.exp(-1.0 / (u * (1.0 - u)))
        return out

    # Build F, theta, x, y (ground truth)
    G_vals = G(u)
    C = trapezoid(G_vals, u)
    F_vals = cumulative_trapezoid(G_vals, u, initial=0.0) / C
    theta_vals = (L / R) * cumulative_trapezoid(F_vals, u, initial=0.0)
    cos_th = np.cos(theta_vals)
    sin_th = np.sin(theta_vals)
    x_vals = L * cumulative_trapezoid(cos_th, u, initial=0.0)
    y_vals = L * cumulative_trapezoid(sin_th, u, initial=0.0)

    kappa = (L / R) * F_vals       # κ(u)
    with np.errstate(divide="ignore"):
        rho = R / F_vals              # radius
    Nx = -np.sin(theta_vals)
    Ny =  np.cos(theta_vals)

    last = -1
    Cx = x_vals[last] + rho[last] * Nx[last]
    Cy = y_vals[last] + rho[last] * Ny[last]
    R_osculating = rho[last]

    print(Cx, Cy)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, y_vals, label=f"Spiral transition (0 ≤ s ≤ {L:0.1f} m)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.add_patch(plt.Circle((Cx, Cy), R_osculating, color="red", fill=False, lw=1, label="Osculating circle"))
    ax.plot([x_vals[last]], [y_vals[last]], "ko", label="Start of Circular Arc")
    ax.plot([x_vals[last], Cx], [y_vals[last], Cy], "k--", alpha=0.6, label="Normal")
    ax.set_ylim(0.0, y_vals[last] * 1.1)
    #ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend()
    plt.show()
    return F_vals, L, N, R, np, plt, s_grid, theta_vals, x_vals, y_vals


@app.cell
def _(F_vals, L, N, R, np, plt, s_grid, theta_vals, x_vals, y_vals):
    # Pick interior expansion point
    s0 = 0.5 * L
    h = s_grid[1] - s_grid[0]
    i0 = np.searchsorted(s_grid, s0)

    # kappa(s) on the grid and its derivatives up to order 8
    kappa_vals = (1.0 / R) * F_vals

    def nth_derivative_on_uniform(f_vals, h, n):
        g = f_vals.copy()
        for _ in range(n):
            g = np.gradient(g, h, edge_order=2)
        return g

    max_kappa_order = 8
    kappa_derivs_grid = [nth_derivative_on_uniform(kappa_vals, h, n)
                         for n in range(max_kappa_order + 1)]
    kappa_at = [kd[i0] for kd in kappa_derivs_grid]  # κ^(n) at s0

    # θ^(n) at s0: θ′=κ, θ″=κ′, ...
    theta0 = theta_vals[i0]
    theta_derivs = [None] * 10  # store θ^(n) for n=0..9 (0 unused for value)
    theta_derivs[1: max_kappa_order + 2] = kappa_at  # shift by 1: θ^(n)=κ^(n-1)

    # x,y values at s0
    x0, y0 = x_vals[i0], y_vals[i0]
    c0, s0_trig = np.cos(theta0), np.sin(theta0)

    # We will compute x^(n), y^(n) at s0 for n=0..9 using recurrences.
    # Initialize arrays of derivatives
    max_order = 9
    x_der = [np.nan] * (max_order + 1)
    y_der = [np.nan] * (max_order + 1)
    x_der[0], y_der[0] = x0, y0
    x_der[1], y_der[1] = c0, s0_trig

    # Helper: compute nth derivative of cos(theta(s)) or sin(theta(s)) at s0
    # using Bell polynomials (Faà di Bruno). We implement iterative product rule
    # by building derivatives of cos(theta) and sin(theta) jointly.
    #
    # Let c = cos θ, s = sin θ.
    # Derivatives satisfy:
    #   d/ds [c] = -θ' s
    #   d/ds [s] =  θ' c
    # and then repeatedly differentiate, using θ^(k) and already-known c^(j), s^(j).
    # We'll build lists c^(n), s^(n) at s0.

    c_der = [np.nan] * (max_order + 1)
    s_der = [np.nan] * (max_order + 1)
    c_der[0], s_der[0] = c0, s0_trig

    for n in range(1, max_order + 1):
        # c^(n) = d/ds c^(n-1)
        # Use Leibniz: derivative of products of theta_derivs and c/s lower orders.
        # Maintain running sums:
        # From base relations:
        # c' = -θ' s
        # s' =  θ' c
        # For higher n, apply d/ds to the right-hand side recursively.
        # We'll compute via explicit sums:
        # c^(n) = - sum_{k=0}^{n-1} binom(n-1,k) θ^(k+1) s^(n-1-k)
        # s^(n) =  sum_{k=0}^{n-1} binom(n-1,k) θ^(k+1) c^(n-1-k)
        # Proof: by repeated differentiation of base and Leibniz; this is a compact, exact identity.
        from math import comb
        cn = 0.0
        sn = 0.0
        for k in range(n):
            tk1 = theta_derivs[k + 1]  # θ^(k+1)
            coef = comb(n - 1, k)
            cn += -coef * tk1 * s_der[n - 1 - k]
            sn +=  coef * tk1 * c_der[n - 1 - k]
        c_der[n] = cn
        s_der[n] = sn

    # Now x^(n), y^(n) recurrences:
    # x' = c, y' = s
    # So x^(n) = c^(n-1), y^(n) = s^(n-1) for n>=1.
    for n in range(2, max_order + 1):
        x_der[n] = c_der[n - 1]
        y_der[n] = s_der[n - 1]

    # Build Taylor partial sums for orders m=0..max_order
    def taylor_series_from_derivs(s, s0, derivs):
        ds = s - s0
        out = np.zeros_like(s)
        fact = 1.0
        power = np.ones_like(s)
        out += derivs[0]  # f(s0)
        for n in range(1, len(derivs)):
            fact *= n
            power *= ds
            out += derivs[n] / fact * power
        return out

    # Compute approximations of order m for m=0..max_order
    x_approxs = []
    y_approxs = []
    for m in range(max_order + 1):
        x_approxs.append(
            taylor_series_from_derivs(s_grid, s0, x_der[: m + 1])
        )
        y_approxs.append(
            taylor_series_from_derivs(s_grid, s0, y_der[: m + 1])
        )

    # Compare on a local window around s0
    w = int(0.14 * N)  # half-width; keep moderate so series converges visually
    lo = max(0, i0 - w)
    hi = min(N, i0 + w)

    # Plot error curves for orders 3..9
    orders = list(range(3, 10))
    plt.figure(figsize=(10, 5))
    for m in orders:
        err = np.hypot(x_approxs[m] - x_vals, (y_approxs[m] - y_vals) * 1000)
        plt.plot(s_grid[lo:hi], err[lo:hi], label=f'{m}th')
    plt.xlabel('s [m]'); 
    plt.ylabel('position error [mm]')
    plt.title('Taylor approximation error (orders 3–9) around s0')
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
