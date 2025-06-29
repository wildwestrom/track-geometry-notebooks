import numpy as np
from scipy.optimize import minimize


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


def optimize_paths(H, start_point, end_point, epsilon, segments):
    """
    Optimize paths for a given terrain and parameters.
    Returns a tuple (optimal_paths, results) where:
      - optimal_paths: list of (x_path, y_path) for each N
      - results: list of dicts with optimization results
    """
    results = []
    for N in range(1, segments + 1):
        print(f"--- Optimizing for N = {N} turns ---")
        # Initial guess
        x0 = []
        for i in range(1, N + 1):
            frac = i / (N + 1)
            vx = start_point[0] * (1 - frac) + end_point[0] * frac
            vy = start_point[1] * (1 - frac) + end_point[1] * frac
            R = 1.0
            w = 0.1
            x0.extend([vx, vy, R, w])
        x0 = np.array(x0)
        # Bounds
        bounds = []
        for i in range(N):
            bounds.extend(
                [
                    (0, 1),  # x_i
                    (0, 1),  # y_i
                    (0.05, 10),  # R_i
                    (0.0, np.pi),  # omega_i
                ]
            )
        # Optimize
        result = minimize(
            fun=objective_function_J,
            x0=x0,
            args=(
                start_point,
                end_point,
                H,
                epsilon,
            ),
            method="SLSQP",
            bounds=bounds,
        )
        print(f"Result for N={N}: Cost = {result.fun:.4f}")
        results.append({
            "cost": result.fun,
            "N": N,
            "x_optimal": result.x,
            "success": result.success,
        })
    optimal_paths = []
    for result in results:
        if result["cost"] < float("inf"):
            x_optimal = result["x_optimal"]
            N = result["N"]
            # Extract vertices
            vertices = []
            for i in range(N):
                x_i = x_optimal[4 * i]
                y_i = x_optimal[4 * i + 1]
                vertices.append((x_i, y_i))
            # Create path for visualization
            path_points = [start_point] + vertices + [end_point]
            x_path = [p[0] for p in path_points]
            y_path = [p[1] for p in path_points]
            optimal_paths.append((x_path, y_path))
        else:
            print("Optimization failed to find a solution")
    return optimal_paths, results
