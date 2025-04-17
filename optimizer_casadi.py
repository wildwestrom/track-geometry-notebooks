import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os
from terrain_gen import generate_terrain
from numpy import ndarray
from skimage.graph import route_through_array
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

# Define physical constants and scales
TERRAIN_SIZE_KM: float = 50.0  # Length of each side in km
MIN_ELEVATION_M: float = 10.0  # m above sea level
MAX_ELEVATION_M: float = 500.0  # m above sea level
GRID_SIZE: int = 128  # Number of grid points in each dimension
DEFAULT_FIGSIZE: tuple[int, int] = (14, 8)
TEST_INITIAL = False


class RailwayPathOptimizer:
    """
    Optimizes railway paths by minimizing a combination of:
    - Curvature (for passenger comfort)
    - Change in curvature (for smoothness)
    - Construction cost
    - Travel time
    """

    def __init__(self):
        """
        Initialize the optimizer.

        Args:
            terrain: A 2D array representing the terrain elevation (required)
        """

        def scale_terrain(terrain, min_elev, max_elev):
            return min_elev + (terrain * (max_elev - min_elev))

        print("Generating terrain...")
        # Generate base terrain (values in range 0.0-1.0)
        terrain = MIN_ELEVATION_M + (
            generate_terrain(
                size=GRID_SIZE,  # Use mostly default settings
                smoothing=1.0,
                seed=42,  # Random seed for reproducibility
            )
            * MAX_ELEVATION_M
            - MIN_ELEVATION_M
        )
        self.terrain = terrain
        print(
            f"Terrain elevation range: {MIN_ELEVATION_M:.1f}m to {MAX_ELEVATION_M:.1f}m"
        )

        self.terrain_cost = self.create_terrain_cost_from_array(
            terrain,
            terrain_size=(TERRAIN_SIZE_KM, TERRAIN_SIZE_KM),
        )

        # Get terrain dimensions
        self.height, self.width = self.terrain.shape
        self.x_bounds = (0, 1)  # Normalized coordinates
        self.y_bounds = (0, 1)  # Normalized coordinates

        self.scale_factors = {
            "distance": TERRAIN_SIZE_KM,  # km
            "elevation": MAX_ELEVATION_M - MIN_ELEVATION_M,  # m
            "min_elevation": MIN_ELEVATION_M,  # m
        }
        self.default_figsize = DEFAULT_FIGSIZE

        # Create terrain interpolant
        x_grid = np.linspace(self.x_bounds[0], self.x_bounds[1], self.width)
        y_grid = np.linspace(self.y_bounds[0], self.y_bounds[1], self.height)
        self.terrain_interpolant = ca.interpolant(
            "terrain_interp", "bspline", [x_grid, y_grid], self.terrain.flatten()
        )

    def create_terrain_cost_from_array(
        self, terrain: ndarray, terrain_size: tuple[float, float] = (1, 1)
    ) -> callable:
        """
        Create a cost function from a terrain array using a direct lookup approach compatible with CasADi.
        Args:
            terrain: A 2D array of terrain heights/costs
            terrain_size: The real-world size of the terrain (width, height)
        Returns:
            A function that provides terrain cost at a given point
        """
        # Inline gradient and cost field computation
        height, width = terrain.shape
        dx = terrain_size[0] / (width - 1)
        dy = terrain_size[1] / (height - 1)
        grad_y, grad_x = np.gradient(terrain, dy, dx)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        cost_field = grad_mag**2
        self.terrain_gradient = (grad_x, grad_y)
        self.terrain_gradient_mag = grad_mag
        self.terrain_cost_field = cost_field
        # Create CasADi interpolant for the cost field
        x_params = np.linspace(0, terrain_size[0], width)
        y_params = np.linspace(0, terrain_size[1], height)
        cost_flat = cost_field.flatten()
        try:
            cost_lookup = ca.interpolant(
                "cost_lookup", "linear", [x_params, y_params], cost_flat
            )
        except:
            print("Warning: Failed to create CasADi cost interpolant. Using fallback.")
            cost_lookup = None

        def cost_function(x, y):
            if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                i = int(min(max(y * (height - 1), 0), height - 1))
                j = int(min(max(x * (width - 1), 0), width - 1))
                return cost_field[i, j]
            elif cost_lookup is not None:
                try:
                    xy = ca.vertcat(x, y)
                    return cost_lookup(xy)
                except:
                    return 1.0 + 0.2 * (x + y)
            else:
                return 1.0 + 0.2 * (x + y)

        return cost_function

    def harsh_gradient_penalty(self, gc):
        """
        Quadratic penalty for |gc| <= 0.04, harsh exponential for |gc| > 0.04.
        Handles both CasADi symbolic and numpy/python float types.
        """
        if isinstance(gc, (ca.MX, ca.SX)):
            abs_gc = ca.fabs(gc)
            return ca.if_else(
                abs_gc <= 0.04,
                10 * gc**2,
                10 * 0.04**2 + (ca.exp(8 * (abs_gc - 0.04)) - 1),
            )
        else:
            abs_gc = abs(gc)
            if abs_gc <= 0.04:
                return 10 * gc**2
            else:
                return 10 * 0.04**2 + (np.exp(8 * (abs_gc - 0.04)) - 1)

    def compute_path_djikstras(
        self,
        start,
        end,
        n_points,
        max_grade=0.04,
    ):
        """
        Compute a geodesic (least-cost) path using Dijkstra's algorithm on a grid,
        penalizing excessive grade using the same logic as the optimizer objective.
        Uses 4-connectivity. Resamples to n_points.
        """
        terrain = self.terrain
        height, width = terrain.shape
        dx = self.scale_factors["distance"] * 1000 / (width - 1)
        dy = self.scale_factors["distance"] * 1000 / (height - 1)
        N = height * width
        adj = lil_matrix((N, N))
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                elev = terrain[i, j]
                for di, dj, dist in [(-1, 0, dy), (1, 0, dy), (0, -1, dx), (0, 1, dx)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        nidx = ni * width + nj
                        elev_n = terrain[ni, nj]
                        dxy = dist
                        dz = elev_n - elev
                        grade = dz / (dxy + 1e-6)
                        penalty = self.harsh_gradient_penalty(grade)
                        cost = dxy + penalty
                        adj[idx, nidx] = cost
        start_idx = int(start[1] * (height - 1)) * width + int(start[0] * (width - 1))
        end_idx = int(end[1] * (height - 1)) * width + int(end[0] * (width - 1))
        _, predecessors = dijkstra(
            csgraph=adj, directed=True, indices=start_idx, return_predecessors=True
        )
        path = []
        cur = end_idx
        while cur != start_idx and cur != -9999:
            path.append(cur)
            cur = predecessors[cur]
        path.append(start_idx)
        path = path[::-1]
        y_path = np.array([p // width for p in path]) / (height - 1)
        x_path = np.array([p % width for p in path]) / (width - 1)
        dists = np.zeros(len(x_path))
        for i in range(1, len(x_path)):
            dists[i] = dists[i - 1] + np.sqrt(
                (x_path[i] - x_path[i - 1]) ** 2 + (y_path[i] - y_path[i - 1]) ** 2
            )
        dists = dists / dists[-1] if dists[-1] > 0 else dists
        x_resampled = np.interp(np.linspace(0, 1, n_points), dists, x_path)
        y_resampled = np.interp(np.linspace(0, 1, n_points), dists, y_path)
        return x_resampled, y_resampled

    def get_initial_guess(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        n_points: int,
        terrain_cost_weight: float,
    ) -> tuple[ndarray, ndarray]:
        if terrain_cost_weight > 0:
            return self.compute_path_djikstras(start_point, end_point, n_points)
        x_init: ndarray[float] = np.linspace(start_point[0], end_point[0], n_points)
        y_init: ndarray[float] = np.linspace(start_point[1], end_point[1], n_points)
        return x_init, y_init

    # --------------------- BEGIN OPTIMIZER --------------------#

    def optimize_path(
        self,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        max_curvature: float = 0.3,
        max_gradient: float = 0.15,
        weights: tuple[float, float, float, float, float, str] = (
            1.0,
            2.0,
            1.0,
            1.0,
            0.5,
            "default weights",
        ),
        n_points: int = 40,
    ) -> tuple[ndarray, ndarray, ndarray]:
        if n_points < 3:
            raise ValueError(
                "n_points must be at least 3 to define curvature and gradients."
            )
        (
            curvature_weight,
            curvature_change_weight,
            gradient_weight,
            terrain_cost_weight,
            time_weight,
            _label,
        ) = weights

        opti = ca.Opti()
        x = opti.variable(n_points)
        y = opti.variable(n_points)
        z = opti.variable(n_points)  # Track elevation (meters)

        # Fix start and end points (horizontal and vertical)
        opti.subject_to(x[0] == start_point[0])
        opti.subject_to(y[0] == start_point[1])
        # Set start/end elevation to terrain at those points
        start_elev = self.terrain_interpolant(
            ca.vertcat(start_point[0], start_point[1])
        )
        end_elev = self.terrain_interpolant(ca.vertcat(end_point[0], end_point[1]))
        opti.subject_to(z[0] == start_elev)
        opti.subject_to(z[-1] == end_elev)
        opti.subject_to(x[-1] == end_point[0])
        opti.subject_to(y[-1] == end_point[1])

        buffer = 0.001
        for i in range(n_points):
            opti.subject_to(x[i] >= self.x_bounds[0] + buffer)
            opti.subject_to(x[i] <= self.x_bounds[1] - buffer)
            opti.subject_to(y[i] >= self.y_bounds[0] + buffer)
            opti.subject_to(y[i] <= self.y_bounds[1] - buffer)

        # Convert normalized x, y to meters for distance calculation
        x_m = x * self.scale_factors["distance"] * 1000
        y_m = y * self.scale_factors["distance"] * 1000
        z_m = z  # already in meters
        ds = []
        for i in range(n_points - 1):
            ds.append(
                ca.sqrt(
                    (x_m[i + 1] - x_m[i]) ** 2
                    + (y_m[i + 1] - y_m[i]) ** 2
                    + (z_m[i + 1] - z_m[i]) ** 2
                )
            )

        # --- Gradient constraints and calculation (on track profile) ---
        gradients = []
        elevation_changes = []
        cumulative_elevation_gain = 0
        for i in range(n_points - 1):
            dz = z[i + 1] - z[i]  # meters
            elevation_changes.append(dz)
            cumulative_elevation_gain += ca.fmax(0, dz)
            dx = x[i + 1] - x[i]
            dy = y[i + 1] - y[i]
            # Use 3D distance for gradient calculation
            dx_m = dx * self.scale_factors["distance"] * 1000
            dy_m = dy * self.scale_factors["distance"] * 1000
            dz_m = dz  # already in meters
            dist_3d = ca.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
            gradient = dz / (dist_3d + 1e-6)
            gradients.append(gradient)
            opti.subject_to(gradient <= max_gradient)
            opti.subject_to(gradient >= -max_gradient)

        # Calculate required average gradient (start to end)
        total_elevation_change = z[-1] - z[0]
        total_distance = sum(ds) * self.scale_factors["distance"] * 1000
        required_gradient = total_elevation_change / (total_distance + 1e-6)

        # Penalize deviation from required average gradient and local changes
        gradient_deviation_obj = (
            gradient_weight
            * sum((g - required_gradient) ** 2 for g in gradients)
            / max(1, len(gradients))
        )
        # Exponential penalty for local gradient changes near ±4%
        gradient_changes = [
            ca.fabs(gradients[i] - gradients[i - 1]) for i in range(1, len(gradients))
        ]

        gradient_change_obj = (
            gradient_weight
            * sum(self.harsh_gradient_penalty(gc) for gc in gradient_changes)
            / max(1, len(gradient_changes))
        )

        # --- Curvature (3D) ---
        curvature = []
        curvature_change = []
        direction_changes = []
        for i in range(1, n_points - 1):
            v1x = x[i] - x[i - 1]
            v1y = y[i] - y[i - 1]
            v1z = z[i] - z[i - 1]
            v2x = x[i + 1] - x[i]
            v2y = y[i + 1] - y[i]
            v2z = z[i + 1] - z[i]
            # 3D vectors
            v1 = ca.vertcat(v1x, v1y, v1z)
            v2 = ca.vertcat(v2x, v2y, v2z)
            # Cross product
            cross_x = v1y * v2z - v1z * v2y
            cross_y = v1z * v2x - v1x * v2z
            cross_z = v1x * v2y - v1y * v2x
            cross_norm = ca.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
            v1_norm = ca.sqrt(v1x**2 + v1y**2 + v1z**2)
            v2_norm = ca.sqrt(v2x**2 + v2y**2 + v2z**2)
            v_sum_x = v1x + v2x
            v_sum_y = v1y + v2y
            v_sum_z = v1z + v2z
            v_sum_norm = ca.sqrt(v_sum_x**2 + v_sum_y**2 + v_sum_z**2)
            denom = v1_norm * v2_norm * v_sum_norm + 1e-8
            k = 2 * cross_norm / denom
            curvature.append(k)
            # For direction change, use angle between v1 and v2 in 3D
            dot_product = v1x * v2x + v1y * v2y + v1z * v2z
            cos_angle = dot_product / (v1_norm * v2_norm + 1e-8)
            cos_angle = ca.fmin(ca.fmax(cos_angle, -1), 1)  # Clamp for safety
            direction_change = ca.acos(cos_angle) * 180 / np.pi
            direction_changes.append(direction_change)
            opti.subject_to(k <= max_curvature)
            if i > 1:
                prev_k = curvature[-2]
                dk = k - prev_k
                curvature_change.append(dk)
                opti.subject_to(ca.fabs(dk) <= 0.15)

        curvature_obj = (
            curvature_weight * sum(k**2 for k in curvature) / max(1, len(curvature))
        )
        curvature_change_obj = (
            curvature_change_weight
            * sum(dk**2 for dk in curvature_change)
            / max(1, len(curvature_change))
        )

        # --- Terrain cost: tunnels, bridges, cuttings, embankments ---
        if terrain_cost_weight == 0:
            terrain_obj = 0
        else:
            tunnel_threshold = -10.0  # meters below terrain
            bridge_threshold = 10.0  # meters above terrain
            bridge_tunnel_multiplier = (
                3.0  # Multiplier to make bridges and tunnels more expensive
            )
            tunnel_cost_per_m = 10000.0 * bridge_tunnel_multiplier
            bridge_cost_per_m = 5000.0 * bridge_tunnel_multiplier
            excavation_cost_per_m3 = 70.0
            embankment_cost_per_m3 = 100.0
            track_width = 10.0  # meters
            terrain_costs = []
            for i in range(n_points - 1):
                terrain_elev1 = self.terrain_interpolant(ca.vertcat(x[i], y[i]))
                terrain_elev2 = self.terrain_interpolant(ca.vertcat(x[i + 1], y[i + 1]))
                track_elev1 = z[i]
                track_elev2 = z[i + 1]
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                dist_m = ca.sqrt(dx**2 + dy**2) * self.scale_factors["distance"] * 1000
                offset1 = track_elev1 - terrain_elev1
                offset2 = track_elev2 - terrain_elev2
                tunnel1 = ca.fmax(0, -(offset1 + tunnel_threshold))
                tunnel2 = ca.fmax(0, -(offset2 + tunnel_threshold))
                tunnel_cost = tunnel_cost_per_m * (tunnel1 + tunnel2) / 2 * dist_m
                bridge1 = ca.fmax(0, offset1 - bridge_threshold)
                bridge2 = ca.fmax(0, offset2 - bridge_threshold)
                bridge_cost = bridge_cost_per_m * (bridge1 + bridge2) / 2 * dist_m
                cut1 = ca.fmax(0, -offset1)
                cut2 = ca.fmax(0, -offset2)
                cutting_cost = (
                    excavation_cost_per_m3 * (cut1 + cut2) / 2 * dist_m * track_width
                )
                fill1 = ca.fmax(0, offset1)
                fill2 = ca.fmax(0, offset2)
                embankment_cost = (
                    embankment_cost_per_m3 * (fill1 + fill2) / 2 * dist_m * track_width
                )
                segment_cost = (
                    tunnel_cost + bridge_cost + cutting_cost + embankment_cost
                )
                terrain_costs.append(segment_cost)
            cost_obj = (
                terrain_cost_weight * sum(terrain_costs) / max(1, len(terrain_costs))
            )

            # --- Gradient Flow Penalty using terrain_cost ---
            gradient_flow_penalty = 0
            for i in range(n_points - 1):
                # Midpoint of segment
                xm = (x[i] + x[i + 1]) / 2
                ym = (y[i] + y[i + 1]) / 2
                # Use terrain_cost as the penalty at the midpoint
                gradient_flow_penalty += self.terrain_cost(xm, ym)
            gradient_flow_penalty = (
                terrain_cost_weight * gradient_flow_penalty / max(1, len(gradients) - 1)
            )
            terrain_obj = (
                cost_obj
                + gradient_deviation_obj
                + gradient_change_obj
                + gradient_flow_penalty
            )

        # --- Improved Travel Time Calculation ---
        # Parameters for train dynamics
        max_speed = 80.0  # m/s (about 288 km/h)
        max_accel = 0.7  # m/s^2 (typical for passenger trains)
        max_decel = 0.7  # m/s^2 (braking, positive value)
        max_lateral_accel = 1.0  # m/s^2 (comfort limit)
        g = 9.81  # m/s^2

        # Compute curvature-limited speed for each segment
        v_curve = [max_speed] * (n_points - 1)
        for i in range(1, n_points - 1):
            k = curvature[i - 1]  # curvature for segment i (already computed)
            v_curve[i - 1] = ca.sqrt(max_lateral_accel / (ca.fabs(k) + 1e-8))
        # Compute gradient-limited speed for each segment (simple traction/braking model)
        v_grade = [max_speed] * (n_points - 1)
        for i in range(n_points - 1):
            g_i = gradients[i]
            # Uphill: limit by available acceleration
            v_grade[i] = ca.if_else(
                g_i > 0, max_speed, ca.sqrt(2 * max_decel / (ca.fabs(g_i) + 1e-8))
            )
        # Segment speed limit is the minimum of all constraints
        v_limit = [ca.fmin(v_curve[i], v_grade[i]) for i in range(n_points - 1)]
        v_limit = [ca.fmin(v_limit[i], max_speed) for i in range(n_points - 1)]

        # Forward-backward pass for speed profile (CasADi symbolic)
        v_fwd = [0] * n_points
        v_fwd[0] = 0  # Start from rest
        for i in range(n_points - 1):
            v_possible = ca.sqrt(v_fwd[i] ** 2 + 2 * max_accel * ds[i])
            v_fwd[i + 1] = ca.fmin(v_possible, v_limit[i])
        v_bwd = [0] * n_points
        v_bwd[-1] = 0  # End at rest
        for i in reversed(range(n_points - 1)):
            v_possible = ca.sqrt(v_bwd[i + 1] ** 2 + 2 * max_decel * ds[i])
            v_bwd[i] = ca.fmin(v_possible, v_limit[i])
        # Actual speed at each segment is the minimum of forward and backward pass
        v_profile = [ca.fmin(v_fwd[i], v_bwd[i]) for i in range(n_points)]
        # Compute travel time for each segment
        travel_times = [ds[i] / (v_profile[i] + 1e-6) for i in range(n_points - 1)]
        total_travel_time = sum(travel_times)
        time_obj = time_weight * total_travel_time

        objective = curvature_obj + curvature_change_obj + terrain_obj + time_obj
        opti.minimize(objective)

        # Initial guess for path
        x_init, y_init = self.get_initial_guess(
            start_point, end_point, n_points, terrain_cost_weight
        )
        # Set z_init to follow the terrain elevation along the initial path
        z_init = np.array(
            [
                float(
                    self.terrain[
                        int(y_init[i] * (self.height - 1)),
                        int(x_init[i] * (self.width - 1)),
                    ]
                )
                for i in range(n_points)
            ]
        )
        opti.set_initial(x, x_init)
        opti.set_initial(y, y_init)
        opti.set_initial(z, z_init)

        # Compute average segment length of initial guess (in meters)
        x_init_m = np.array(x_init) * self.scale_factors["distance"] * 1000
        y_init_m = np.array(y_init) * self.scale_factors["distance"] * 1000
        z_init_m = np.array(z_init)  # already in meters
        avg_segment_length = np.mean(
            [
                np.sqrt(
                    (x_init_m[i + 1] - x_init_m[i]) ** 2
                    + (y_init_m[i + 1] - y_init_m[i]) ** 2
                    + (z_init_m[i + 1] - z_init_m[i]) ** 2
                )
                for i in range(n_points - 1)
            ]
        )

        # Add segment length constraints based on initial path (in meters)
        for i in range(n_points - 1):
            opti.subject_to(ds[i] >= 0.8 * avg_segment_length)
            opti.subject_to(ds[i] <= 3.0 * avg_segment_length)

        # Add minimum total path length constraint (at least 90% of initial guess)
        total_initial_length = np.sum(
            [
                np.sqrt(
                    (x_init_m[i + 1] - x_init_m[i]) ** 2
                    + (y_init_m[i + 1] - y_init_m[i]) ** 2
                    + (z_init_m[i + 1] - z_init_m[i]) ** 2
                )
                for i in range(n_points - 1)
            ]
        )
        opti.subject_to(sum(ds) >= 0.8 * total_initial_length)

        options = {
            "ipopt": {
                "max_iter": 0 if TEST_INITIAL else 500,
                "tol": 1e-2,
                "acceptable_tol": 1e-1,
                "mu_strategy": "adaptive",
                "hessian_approximation": "limited-memory",
                "linear_solver": "mumps",
                "limited_memory_max_history": 50,
                "bound_push": 0.01,
                "bound_frac": 0.01,
                "warm_start_init_point": "yes",
                "print_level": 5,
                "nlp_scaling_method": "gradient-based",
                "alpha_for_y": "safer-min-dual-infeas",
                "recalc_y": "yes",
                "acceptable_iter": 10,
                "acceptable_obj_change_tol": 1e-2,
                "constr_viol_tol": 1e-4,
            }
        }
        opti.solver("ipopt", options)

        try:
            sol = opti.solve()
            x_coords = sol.value(x)
            y_coords = sol.value(y)
            z_coords = sol.value(z)
            get_val = sol.value
        except Exception as e:
            print(f"Optimization failed: {e}")
            try:
                x_coords = opti.debug.value(x)
                y_coords = opti.debug.value(y)
                z_coords = opti.debug.value(z)
                get_val = opti.debug.value
                print("Returning partial solution from the latest solver iteration.")
            except:
                return None, None, None
        # Print the value of each objective term (works for both success and failure)
        print("Objective breakdown:")
        print(f"  curvature_obj:        {get_val(curvature_obj):.4f}")
        print(f"  curvature_change_obj: {get_val(curvature_change_obj):.4f}")
        print(f"  gradient_deviation_obj: {get_val(gradient_deviation_obj):.4f}")
        print(f"  gradient_change_obj:  {get_val(gradient_change_obj):.4f}")
        print(f"  terrain_obj:          {get_val(terrain_obj):.4f}")
        print(f"  path_length:          {get_val(total_distance / 1000):.4f}")
        print(f"  time_obj:             {get_val(time_obj):.4f}")
        print(f"  TOTAL OBJECTIVE:      {get_val(objective):.4f}")
        return x_coords, y_coords, z_coords

    # ---------------------- END OPTIMIZER ---------------------#

    def plot_path(self, coords: ndarray, title: str = "Optimized Railway Path"):
        """
        Plot the optimized path with real-world units, showing both terrain and track elevation profiles.
        coords: numpy array of shape (N, 3) where columns are x, y, z
        """
        terrain = self.terrain
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]
        fig, ax = plt.subplots(figsize=self.default_figsize)
        # Convert coordinates to kilometers
        x_km = x_coords * self.scale_factors["distance"]
        y_km = y_coords * self.scale_factors["distance"]
        # Plot points with enhanced visibility
        ax.plot(
            x_km[0],
            y_km[0],
            "bo",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
            label="Start",
        )
        ax.plot(
            x_km[-1],
            y_km[-1],
            "go",
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=2,
            zorder=5,
            label="End",
        )
        if len(x_km) > 2:
            ax.plot(
                x_km[1:-1],
                y_km[1:-1],
                "ro",
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=1,
                alpha=0.7,
                zorder=4,
            )
        # Plot terrain with appropriate opacity
        if terrain is not None:
            extent_km = (
                0,
                self.scale_factors["distance"],
                0,
                self.scale_factors["distance"],
            )
            terrain_plot = ax.imshow(
                terrain,
                extent=extent_km,
                origin="lower",
                alpha=0.6,
                cmap=plt.cm.terrain,
                zorder=1,
            )
            plt.colorbar(
                terrain_plot, ax=ax, label="Elevation (meters above sea level)"
            )
            # Overlay contour lines (in meters)
            contour_levels = np.linspace(np.min(terrain), np.max(terrain), 20)
            contour = ax.contour(
                np.linspace(0, self.scale_factors["distance"], terrain.shape[1]),
                np.linspace(0, self.scale_factors["distance"], terrain.shape[0]),
                terrain,
                levels=contour_levels,
                colors="black",
                alpha=0.2,
                linewidths=0.5,
                zorder=2,
            )
            plt.clabel(contour, inline=True, fontsize=8, fmt="%.0fm")
        # Calculate gradient for coloring (track profile)
        gradients = []
        segments = []
        for i in range(len(x_km) - 1):
            dx = x_km[i + 1] - x_km[i]  # km
            dy = y_km[i + 1] - y_km[i]  # km
            dz = z_coords[i + 1] - z_coords[i]  # meters
            distance_m = ca.sqrt(dx**2 + dy**2) * 1000  # convert km to m
            gradient = dz / (distance_m + 1e-6)  # m/m (dimensionless)
            gradients.append(gradient)
            segments.append([(x_km[i], y_km[i]), (x_km[i + 1], y_km[i + 1])])
        if segments:
            max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
            norm = plt.Normalize(-max_abs_gradient, max_abs_gradient)
            # Red (up), White (zero), Blue (down)
            colors = [
                (0.0, 0.2, 0.8),
                (1.0, 1.0, 1.0),
                (0.8, 0.2, 0.8),
            ]  # Blue -> White -> Red
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list("gradient", colors, N=n_bins)
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
            lc.set_array(np.array(gradients))
            ax.add_collection(lc)
        ax.legend(loc="lower right", framealpha=1, facecolor="white", edgecolor="black")
        # Calculate statistics
        total_length_km = (
            sum(
                ca.sqrt(
                    (x_coords[i + 1] - x_coords[i]) ** 2
                    + (y_coords[i + 1] - y_coords[i]) ** 2
                )
                for i in range(len(x_coords) - 1)
            )
            * self.scale_factors["distance"]
        )
        max_gradient = max(abs(g) for g in gradients) if gradients else 0
        avg_gradient = (
            sum(abs(g) for g in gradients) / max(1, len(gradients)) if gradients else 0
        )
        # Terrain and track elevation stats
        if terrain is not None:
            terrain_elevs = []
            for i in range(len(x_coords)):
                height, width = terrain.shape
                x_idx = int(min(max(x_coords[i] * (width - 1), 0), width - 1))
                y_idx = int(min(max(y_coords[i] * (height - 1), 0), height - 1))
                terrain_elevs.append(terrain[y_idx, x_idx])
            avg_terrain_elev = sum(terrain_elevs) / max(1, len(terrain_elevs))
            max_terrain_elev = max(terrain_elevs)
            min_terrain_elev = min(terrain_elevs)
        else:
            avg_terrain_elev = max_terrain_elev = min_terrain_elev = 0
        avg_track_elev = np.mean(z_coords)
        max_track_elev = np.max(z_coords)
        min_track_elev = np.min(z_coords)
        stats_text = (
            f"Path Length: {total_length_km:.2f} km\n"
            f"Max Gradient: {max_gradient:.1%}\n"
            f"Avg Gradient: {avg_gradient:.1%}\n"
            f"Min Terrain Elev: {min_terrain_elev:.0f} m\n"
            f"Avg Terrain Elev: {avg_terrain_elev:.0f} m\n"
            f"Max Terrain Elev: {max_terrain_elev:.0f} m\n"
            f"Min Track Elev: {min_track_elev:.0f} m\n"
            f"Avg Track Elev: {avg_track_elev:.0f} m\n"
            f"Max Track Elev: {max_track_elev:.0f} m"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        ax.set_title(title)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Distance (km)")
        ax.grid(True, alpha=0.3, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlim(0, self.scale_factors["distance"])
        ax.set_ylim(0, self.scale_factors["distance"])
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        config_name = title.replace(" Configuration", "")
        config_name = config_name.replace(" ", "_").lower()
        filename = f"plots/{config_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {filename}")
        plt.show()

    def plot_combined_profile(
        self, coords: ndarray, title: str = "Railway Path Profile"
    ):
        """
        Create a combined plot showing both terrain and track elevation profiles, and the track gradient.
        coords: numpy array of shape (N, 3) where columns are x, y, z
        """
        terrain = self.terrain
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        z_coords = coords[:, 2]
        x_km = x_coords * self.scale_factors["distance"]
        y_km = y_coords * self.scale_factors["distance"]
        distances = [0]
        gradients = []
        terrain_elevations = []
        track_elevations = []
        for i in range(len(x_km) - 1):
            dx = x_km[i + 1] - x_km[i]
            dy = y_km[i + 1] - y_km[i]
            segment_distance = ca.sqrt(dx**2 + dy**2)
            if segment_distance < 1e-6:
                continue
            distances.append(distances[-1] + segment_distance)
            height, width = terrain.shape
            x_idx = int(min(max(x_coords[i] * (width - 1), 0), width - 1))
            y_idx = int(min(max(y_coords[i] * (height - 1), 0), height - 1))
            terrain_elev = terrain[y_idx, x_idx]
            terrain_elevations.append(terrain_elev)
            track_elev = z_coords[i]
            track_elevations.append(track_elev)
            dz = z_coords[i + 1] - z_coords[i]
            distance_m = segment_distance * 1000
            gradient = dz / (distance_m + 1e-6)
            gradients.append(gradient)
        # Add last point
        if len(x_km) > 0:
            height, width = terrain.shape
            x_idx = int(min(max(x_coords[-1] * (width - 1), 0), width - 1))
            y_idx = int(min(max(y_coords[-1] * (height - 1), 0), height - 1))
            terrain_elev = terrain[y_idx, x_idx]
            terrain_elevations.append(terrain_elev)
            track_elev = z_coords[-1]
            track_elevations.append(track_elev)
        # Filter anomalous gradients
        if gradients:
            gradient_array = np.array(gradients)
            mean_gradient = np.mean(gradient_array)
            std_gradient = np.std(gradient_array)
            threshold = 3 * std_gradient
            for i in range(len(gradients)):
                if abs(gradients[i] - mean_gradient) > threshold:
                    prev_valid = next_valid = None
                    prev_idx = i - 1
                    next_idx = i + 1
                    while prev_idx >= 0:
                        if abs(gradients[prev_idx] - mean_gradient) <= threshold:
                            prev_valid = gradients[prev_idx]
                            break
                        prev_idx -= 1
                    while next_idx < len(gradients):
                        if abs(gradients[next_idx] - mean_gradient) <= threshold:
                            next_valid = gradients[next_idx]
                            break
                        next_idx += 1
                    if prev_valid is not None and next_valid is not None:
                        gradients[i] = (prev_valid + next_valid) / 2
                    elif prev_valid is not None:
                        gradients[i] = prev_valid
                    elif next_valid is not None:
                        gradients[i] = next_valid
                    else:
                        gradients[i] = mean_gradient
        # Compute midpoints for gradient plotting
        gradient_midpoints = [
            (distances[i] + distances[i + 1]) / 2 for i in range(len(gradients))
        ]
        # For N points, there are N-1 segments and thus N-1 gradients.
        # For each segment, repeat the left and right endpoint, and the gradient value.
        step_x = []
        step_y = []
        for i in range(len(gradients)):
            step_x.extend([distances[i], distances[i + 1]])
            step_y.extend([gradients[i], gradients[i]])
        step_x = np.array(step_x)
        step_y = np.array(step_y)
        fig, ax1 = plt.subplots(figsize=self.default_figsize)
        color_terrain = "gray"
        color_track = "green"
        ax1.set_xlabel("Distance along path (km)")
        ax1.set_ylabel("Elevation (m)")
        ax1.plot(
            distances,
            terrain_elevations,
            "--",
            color=color_terrain,
            linewidth=1.5,
            label="Terrain Elevation",
        )
        ax1.plot(
            distances,
            track_elevations,
            "-",
            color=color_track,
            linewidth=2.5,
            label="Track Elevation",
        )
        ax1.tick_params(axis="y")
        min_elev = min(min(terrain_elevations), min(track_elevations))
        max_elev = max(max(terrain_elevations), max(track_elevations))
        avg_elev = np.mean(track_elevations)
        ax2 = ax1.twinx()
        color_gradient = "blue"
        ax2.set_ylabel("Gradient (m/m)", color=color_gradient)
        # Plot the block plot for gradient, aligned with each segment
        ax2.plot(step_x, step_y, color="purple", linewidth=1.5, label="Gradient")
        # Fill between for uphill/downhill using block-aligned arrays
        ax2.fill_between(
            step_x,
            step_y,
            0,
            where=step_y > 0,
            color="red",
            alpha=0.3,
            label="Uphill",
        )
        ax2.fill_between(
            step_x,
            step_y,
            0,
            where=step_y <= 0,
            color="blue",
            alpha=0.3,
            label="Downhill",
        )
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax2.tick_params(axis="y", labelcolor=color_gradient)
        ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        elev_range = max_elev - min_elev
        elev_buffer = 1.0 if elev_range < 1 else elev_range * 0.1
        ax1.set_ylim(min_elev - elev_buffer, max_elev + elev_buffer)
        if gradients:
            max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
            gradient_buffer = max_abs_gradient * 0.1
            ax2.set_ylim(
                -max_abs_gradient - gradient_buffer,
                max_abs_gradient + gradient_buffer,
            )
        max_gradient = max(abs(g) for g in gradients) if gradients else 0
        avg_gradient = (
            sum(abs(g) for g in gradients) / max(1, len(gradients)) if gradients else 0
        )
        stats_text = (
            f"Path Length: {distances[-1]:.2f} km\n"
            f"Max Gradient: {max_gradient:.1%}\n"
            f"Avg Gradient: {avg_gradient:.1%}\n"
            f"Min Terrain Elev: {min(terrain_elevations):.1f} m\n"
            f"Avg Track Elev: {avg_elev:.1f} m\n"
            f"Max Track Elev: {max(track_elevations):.1f} m"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.7)
        ax1.text(
            0.05,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        # Update legend to match new plotting
        lines = [ax1.get_lines()[0], ax1.get_lines()[1]] + ax2.lines
        labels = ["Terrain Elevation", "Track Elevation", "Gradient"]
        # Add proxy artists for fill_between
        from matplotlib.patches import Patch

        lines += [
            Patch(facecolor="red", alpha=0.3),
            Patch(facecolor="blue", alpha=0.3),
        ]
        labels += ["Uphill", "Downhill"]
        fig.legend(
            lines, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01)
        )
        plt.subplots_adjust(bottom=0.15)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        os.makedirs("plots", exist_ok=True)
        config_name = title.replace("Railway Path ", "").replace(" Profile", "")
        config_name = config_name.replace(" ", "_").lower()
        filename = f"plots/combined_profile_{config_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved combined profile to {filename}")
        plt.show()
