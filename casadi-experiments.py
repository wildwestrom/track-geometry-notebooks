import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import casadi as ca
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from terrain_gen import generate_terrain

    # Define physical constants and scales
    TERRAIN_SIZE_KM = 50.0 # Length of each side in km
    MIN_ELEVATION_M = 10.0  # 10m above sea level
    MAX_ELEVATION_M = 500.0  # 500m above sea level
    GRID_SIZE = 100  # Number of grid points in each dimension

    class RailwayPathOptimizer:
        """
        Optimizes railway paths by minimizing a combination of:
        - Curvature (for passenger comfort)
        - Change in curvature (for smoothness)
        - Construction cost
        - Travel time
        """

        def __init__(self, terrain=None, terrain_cost=None):
            """
            Initialize the optimizer.

            Args:
                terrain: A 2D array representing the terrain elevation
                terrain_cost: A function that returns the cost of construction at a given point
            """
            self.terrain = terrain
            self.terrain_cost = terrain_cost or (lambda x, y: 0)

            # Get terrain dimensions if provided
            if terrain is not None:
                self.height, self.width = terrain.shape
                self.x_bounds = (0, 1)  # Normalized coordinates
                self.y_bounds = (0, 1)  # Normalized coordinates
            else:
                self.height = self.width = None
                self.x_bounds = self.y_bounds = None

            self.scale_factors = {
                'distance': TERRAIN_SIZE_KM,  # km
                'elevation': MAX_ELEVATION_M - MIN_ELEVATION_M,  # m
                'min_elevation': MIN_ELEVATION_M  # m
            }

            # Create terrain interpolant if terrain is provided
            if terrain is not None:
                x_grid = np.linspace(self.x_bounds[0], self.x_bounds[1], self.width)
                y_grid = np.linspace(self.y_bounds[0], self.y_bounds[1], self.height)
                self.terrain_interpolant = ca.interpolant('terrain_interp', 'bspline', 
                                                        [x_grid, y_grid], 
                                                        terrain.flatten())

        def create_terrain_cost_from_array(self, terrain, terrain_size=(10, 10)):
            """
            Create a cost function from a terrain array using a direct lookup approach compatible with CasADi.

            Args:
                terrain: A 2D array of terrain heights/costs
                terrain_size: The real-world size of the terrain (width, height)

            Returns:
                A function that provides terrain cost at a given point
            """
            height, width = terrain.shape
            x_scale = width / terrain_size[0]
            y_scale = height / terrain_size[1]

            # Flatten the terrain for easier lookup with CasADi
            terrain_flat = terrain.flatten()

            # Create CasADi lookup table function
            # For online use in the optimizer - only used for symbolic evaluation
            try:
                # Pre-compute parameter vectors for the lookup
                x_params = np.linspace(0, terrain_size[0], width)
                y_params = np.linspace(0, terrain_size[1], height)

                # Set up grid for 2D linear interpolation
                grid_x, grid_y = np.meshgrid(x_params, y_params)
                points = np.column_stack([grid_x.flatten(), grid_y.flatten()])

                # Create lookup function
                terrain_lookup = ca.interpolant('terrain_lookup', 'linear', 
                                                [x_params, y_params], 
                                                terrain_flat)
            except:
                # Fallback in case CasADi interpolant creation fails
                print("Warning: Failed to create CasADi interpolant. Using simplified terrain model.")
                terrain_lookup = None

            def cost_function(x, y):
                # For numeric inputs, directly access the array
                if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                    # Convert real-world coordinates to array indices
                    i = int(min(max(y * y_scale, 0), height-1))
                    j = int(min(max(x * x_scale, 0), width-1))
                    return terrain[i, j]
                elif terrain_lookup is not None:
                    # For symbolic variables, use the CasADi interpolant
                    try:
                        # CasADi requires column vectors
                        xy = ca.vertcat(x, y)
                        return terrain_lookup(xy)
                    except:
                        # Fallback if the interpolant fails
                        print("Warning: Interpolant call failed. Using simple terrain model.")
                        return 1.0 + 0.2 * (x + y)  # Simple fallback
                else:
                    # Simple fallback if no interpolant is available
                    return 1.0 + 0.2 * (x + y)

            return cost_function

        def optimize_path(self, start_point, end_point, via_points=None, 
                            max_curvature=0.3,
                            max_gradient=0.15,
                            weights=(1.0, 2.0, 1.0, 0.5), 
                            n_points=40,
                            initial_path=None):
            """
            Optimize a railway path between two points, now with vertical profile.
            """
            opti = ca.Opti()

            # Path coordinates
            x = opti.variable(n_points)
            y = opti.variable(n_points)
            z = opti.variable(n_points)  # Track elevation (meters)

            # Unpack weights
            curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight = weights

            # Fix start and end points (horizontal and vertical)
            opti.subject_to(x[0] == start_point[0])
            opti.subject_to(y[0] == start_point[1])
            # Set start/end elevation to terrain at those points
            start_elev = self.terrain_interpolant(ca.vertcat(start_point[0], start_point[1]))
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

            max_step = 0.2
            min_step = 0.01
            for i in range(n_points-1):
                step_size = ca.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
                opti.subject_to(step_size <= max_step)
                opti.subject_to(step_size >= min_step)  # Enforce minimum step size
                if i < n_points-2:
                    next_step = ca.sqrt((x[i+2]-x[i+1])**2 + (y[i+2]-y[i+1])**2)
                    opti.subject_to(ca.fabs(step_size - next_step) <= 0.08)

            ds = []
            for i in range(n_points-1):
                ds.append(ca.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2))

            boundary_violation = 0
            for i in range(n_points):
                boundary_violation += ca.fmax(0, self.x_bounds[0] - x[i])
                boundary_violation += ca.fmax(0, x[i] - self.x_bounds[1])
                boundary_violation += ca.fmax(0, self.y_bounds[0] - y[i])
                boundary_violation += ca.fmax(0, y[i] - self.y_bounds[1])

            # --- Gradient constraints and calculation (on track profile) ---
            gradients = []
            elevation_changes = []
            cumulative_elevation_gain = 0
            for i in range(n_points-1):
                dz = z[i+1] - z[i]  # meters
                elevation_changes.append(dz)
                cumulative_elevation_gain += ca.fmax(0, dz)
                dx = (x[i+1] - x[i])
                dy = (y[i+1] - y[i])
                dist = ca.sqrt(dx**2 + dy**2) * self.scale_factors['distance'] * 1000  # meters
                gradient = dz / (dist + 1e-6)
                gradients.append(gradient)
                opti.subject_to(gradient <= max_gradient)
                opti.subject_to(gradient >= -max_gradient)

            # --- Curvature (3D) ---
            curvature = []
            curvature_change = []
            direction_changes = []
            for i in range(1, n_points-1):
                v1x = x[i] - x[i-1]
                v1y = y[i] - y[i-1]
                v1z = z[i] - z[i-1]
                v2x = x[i+1] - x[i]
                v2y = y[i+1] - y[i]
                v2z = z[i+1] - z[i]
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
                dot_product = v1x*v2x + v1y*v2y + v1z*v2z
                cos_angle = dot_product / (v1_norm * v2_norm + 1e-8)
                cos_angle = ca.fmin(ca.fmax(cos_angle, -1), 1)  # Clamp for safety
                direction_change = ca.acos(cos_angle) * 180 / np.pi
                direction_changes.append(direction_change)
                opti.subject_to(k <= max_curvature)
                if i > 1:
                    prev_k = curvature[-2]
                    dk = (k - prev_k)
                    curvature_change.append(dk)
                    opti.subject_to(ca.fabs(dk) <= 0.15)

            curvature_obj = curvature_weight * sum(k**2 for k in curvature) / len(curvature)
            curvature_change_obj = curvature_change_weight * sum(dk**2 for dk in curvature_change) / max(1, len(curvature_change))
            gradient_obj = gradient_weight * sum(g**2 for g in gradients) / len(gradients)
            gradient_changes = [ca.fabs(gradients[i] - gradients[i-1]) for i in range(1, len(gradients))]
            gradient_change_obj = 0.5 * gradient_weight * sum(gc**2 for gc in gradient_changes) / max(1, len(gradient_changes))

            # --- Terrain cost: tunnels, bridges, cuttings, embankments ---
            tunnel_threshold = -10.0  # meters below terrain
            bridge_threshold = 10.0   # meters above terrain
            tunnel_cost_per_m = 1000.0
            bridge_cost_per_m = 500.0
            excavation_cost_per_m3 = 50.0
            embankment_cost_per_m3 = 30.0
            track_width = 10.0  # meters
            terrain_costs = []
            for i in range(n_points-1):
                terrain_elev1 = self.terrain_interpolant(ca.vertcat(x[i], y[i]))
                terrain_elev2 = self.terrain_interpolant(ca.vertcat(x[i+1], y[i+1]))
                track_elev1 = z[i]
                track_elev2 = z[i+1]
                dx = (x[i+1] - x[i])
                dy = (y[i+1] - y[i])
                dist_m = ca.sqrt(dx**2 + dy**2) * self.scale_factors['distance'] * 1000
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
                cutting_cost = excavation_cost_per_m3 * (cut1 + cut2) / 2 * dist_m * track_width
                fill1 = ca.fmax(0, offset1)
                fill2 = ca.fmax(0, offset2)
                embankment_cost = embankment_cost_per_m3 * (fill1 + fill2) / 2 * dist_m * track_width
                segment_cost = tunnel_cost + bridge_cost + cutting_cost + embankment_cost
                terrain_costs.append(segment_cost)
            cost_obj = terrain_cost_weight * sum(terrain_costs) / len(terrain_costs)
            path_length = sum(ds) * self.scale_factors['distance']  # in km
            time_obj = time_weight * path_length
            elevation_gain_obj = 0.2 * terrain_cost_weight * cumulative_elevation_gain / self.scale_factors['distance']

            objective = (curvature_obj + curvature_change_obj + 
                       gradient_obj + gradient_change_obj + 
                       cost_obj + elevation_gain_obj + 
                       time_obj + boundary_violation)
            opti.minimize(objective)

            # Initial guess
            if initial_path is not None:
                x_init, y_init = initial_path
            else:
                x_init = np.linspace(start_point[0], end_point[0], n_points)
                y_init = np.linspace(start_point[1], end_point[1], n_points)
            z_init = np.zeros(n_points)
            for i in range(n_points):
                x_val = min(max(x_init[i], 0), 1)
                y_val = min(max(y_init[i], 0), 1)
                x_idx = int(x_val * (self.width - 1))
                y_idx = int(y_val * (self.height - 1))
                z_init[i] = float(self.terrain[y_idx, x_idx])
            opti.set_initial(x, x_init)
            opti.set_initial(y, y_init)
            opti.set_initial(z, z_init)

            options = {
                #"print_time": False,
                "ipopt": {
                    "max_iter": 1000,
                    "tol": 1e-2,
                    "acceptable_tol": 1e-1,
                    "mu_strategy": "adaptive",
                    "hessian_approximation": "limited-memory",
                    "limited_memory_max_history": 50,
                    "bound_push": 0.01,
                    "bound_frac": 0.01,
                    "warm_start_init_point": "yes",
                    #"print_level": 3,
                    "nlp_scaling_method": "gradient-based",
                    "alpha_for_y": "safer-min-dual-infeas",
                    "recalc_y": "yes",
                    "acceptable_iter": 10,
                    "acceptable_obj_change_tol": 1e-2,
                    "constr_viol_tol": 1e-4
                }
            }
            opti.solver('ipopt', options)
            try:
                sol = opti.solve()
                x_coords = sol.value(x)
                y_coords = sol.value(y)
                z_coords = sol.value(z)
                # Print the value of each objective term
                print("Objective breakdown:")
                print(f"  curvature_obj:        {sol.value(curvature_obj):.4f}")
                print(f"  curvature_change_obj: {sol.value(curvature_change_obj):.4f}")
                print(f"  gradient_obj:         {sol.value(gradient_obj):.4f}")
                print(f"  gradient_change_obj:  {sol.value(gradient_change_obj):.4f}")
                print(f"  cost_obj:             {sol.value(cost_obj):.4f}")
                print(f"  elevation_gain_obj:   {sol.value(elevation_gain_obj):.4f}")
                print(f"  time_obj:             {sol.value(time_obj):.4f}")
                print(f"  boundary_violation:   {sol.value(boundary_violation):.4f}")
                print(f"  TOTAL OBJECTIVE:      {sol.value(objective):.4f}")
                return x_coords, y_coords, z_coords
            except Exception as e:
                print(f"Optimization failed: {e}")
                try:
                    x_coords = opti.debug.value(x)
                    y_coords = opti.debug.value(y)
                    z_coords = opti.debug.value(z)
                    print("Returning partial solution from the latest solver iteration.")
                    return x_coords, y_coords, z_coords
                except:
                    return None, None, None

        def plot_path(self, x_coords, y_coords, terrain=None, z_coords=None, title="Optimized Railway Path"):
            """
            Plot the optimized path with real-world units, showing both terrain and track elevation profiles if available.
            """
            fig, ax = plt.subplots(figsize=(8, 5))

            # Convert coordinates to kilometers
            x_km = x_coords * self.scale_factors['distance']
            y_km = y_coords * self.scale_factors['distance']

            # Plot points with enhanced visibility
            ax.plot(x_km[0], y_km[0], 'bo', markersize=10, markeredgecolor='white', 
                   markeredgewidth=2, zorder=5, label='Start')
            ax.plot(x_km[-1], y_km[-1], 'go', markersize=10, markeredgecolor='white', 
                   markeredgewidth=2, zorder=5, label='End')

            if len(x_km) > 2:
                ax.plot(x_km[1:-1], y_km[1:-1], 'ro', markersize=5,
                       markeredgecolor='white', markeredgewidth=1, 
                       alpha=0.7, zorder=4)

            # Plot terrain with appropriate opacity
            if terrain is not None:
                extent_km = (0, self.scale_factors['distance'], 
                           0, self.scale_factors['distance'])

                terrain_plot = ax.imshow(terrain, extent=extent_km, origin='lower', 
                                       alpha=0.6, cmap=plt.cm.terrain, zorder=1)
                plt.colorbar(terrain_plot, ax=ax, label='Elevation (meters above sea level)')

                # Overlay contour lines (in meters)
                contour_levels = np.linspace(np.min(terrain), np.max(terrain), 20)
                contour = ax.contour(
                    np.linspace(0, self.scale_factors['distance'], terrain.shape[1]),
                    np.linspace(0, self.scale_factors['distance'], terrain.shape[0]),
                    terrain,
                    levels=contour_levels,
                    colors='black',
                    alpha=0.2,
                    linewidths=0.5,
                    zorder=2
                )
                plt.clabel(contour, inline=True, fontsize=8, fmt='%.0fm')

            # Calculate gradient for coloring (track profile)
            gradients = []
            segments = []
            for i in range(len(x_km)-1):
                dx = x_km[i+1] - x_km[i]  # km
                dy = y_km[i+1] - y_km[i]  # km
                if z_coords is not None:
                    dz = z_coords[i+1] - z_coords[i]  # meters
                else:
                    dz = 0
                distance_m = np.sqrt(dx**2 + dy**2) * 1000  # convert km to m
                gradient = dz / (distance_m + 1e-6)  # m/m (dimensionless)
                gradients.append(gradient)
                segments.append([(x_km[i], y_km[i]), (x_km[i+1], y_km[i+1])])

            if segments:
                max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
                norm = plt.Normalize(-max_abs_gradient, max_abs_gradient)
                # Red (down), White (zero), Blue (up)
                colors = [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.2, 0.8)]  # Red -> White -> Blue
                n_bins = 256
                cmap = LinearSegmentedColormap.from_list("gradient", colors, N=n_bins)
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
                lc.set_array(np.array(gradients))
                ax.add_collection(lc)

            ax.legend(loc='lower right', framealpha=1, facecolor='white', edgecolor='black')

            # Calculate statistics
            total_length_km = sum(np.sqrt((x_coords[i+1]-x_coords[i])**2 + 
                                        (y_coords[i+1]-y_coords[i])**2)
                                for i in range(len(x_coords)-1)) * self.scale_factors['distance']
            max_gradient = max(abs(g) for g in gradients) if gradients else 0
            avg_gradient = sum(abs(g) for g in gradients)/len(gradients) if gradients else 0

            # Terrain and track elevation stats
            if terrain is not None:
                terrain_elevs = []
                for i in range(len(x_coords)):
                    height, width = terrain.shape
                    x_idx = int(min(max(x_coords[i] * (width - 1), 0), width-1))
                    y_idx = int(min(max(y_coords[i] * (height - 1), 0), height-1))
                    terrain_elevs.append(terrain[y_idx, x_idx])
                avg_terrain_elev = sum(terrain_elevs) / len(terrain_elevs)
                max_terrain_elev = max(terrain_elevs)
                min_terrain_elev = min(terrain_elevs)
            else:
                avg_terrain_elev = max_terrain_elev = min_terrain_elev = 0

            if z_coords is not None:
                avg_track_elev = np.mean(z_coords)
                max_track_elev = np.max(z_coords)
                min_track_elev = np.min(z_coords)
            else:
                avg_track_elev = max_track_elev = min_track_elev = 0

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
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

            ax.set_title(title)
            ax.set_xlabel('Distance (km)')
            ax.set_ylabel('Distance (km)')
            ax.grid(True, alpha=0.3, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(0, self.scale_factors['distance'])
            ax.set_ylim(0, self.scale_factors['distance'])
            plt.tight_layout()
            import os
            os.makedirs('plots', exist_ok=True)
            config_name = title.replace(" Configuration", "")
            config_name = config_name.replace(" ", "_").lower()
            filename = f'plots/{config_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filename}")
            plt.show()

        def plot_combined_profile(self, x_coords, y_coords, terrain=None, z_coords=None, title="Railway Path Profile"):
            """
            Create a combined plot showing both terrain and track elevation profiles, and the track gradient.
            """
            if terrain is None or z_coords is None:
                print("Cannot plot combined profile without terrain and track elevation data")
                return
            x_km = x_coords * self.scale_factors['distance']
            y_km = y_coords * self.scale_factors['distance']
            distances = [0]
            gradients = []
            terrain_elevations = []
            track_elevations = []
            for i in range(len(x_km)-1):
                dx = x_km[i+1] - x_km[i]
                dy = y_km[i+1] - y_km[i]
                segment_distance = np.sqrt(dx**2 + dy**2)
                if segment_distance < 1e-6:
                    continue
                distances.append(distances[-1] + segment_distance)
                height, width = terrain.shape
                x_idx = int(min(max(x_coords[i] * (width - 1), 0), width-1))
                y_idx = int(min(max(y_coords[i] * (height - 1), 0), height-1))
                terrain_elev = terrain[y_idx, x_idx]
                terrain_elevations.append(terrain_elev)
                track_elev = z_coords[i]
                track_elevations.append(track_elev)
                dz = z_coords[i+1] - z_coords[i]
                distance_m = segment_distance * 1000
                gradient = dz / (distance_m + 1e-6)
                gradients.append(gradient)
            # Add last point
            if len(x_km) > 0:
                height, width = terrain.shape
                x_idx = int(min(max(x_coords[-1] * (width - 1), 0), width-1))
                y_idx = int(min(max(y_coords[-1] * (height - 1), 0), height-1))
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
            fig, ax1 = plt.subplots(figsize=(12, 6))
            color_terrain = 'gray'
            color_track = 'green'
            ax1.set_xlabel('Distance along path (km)')
            ax1.set_ylabel('Elevation (m)')
            ax1.plot(distances, terrain_elevations, '--', color=color_terrain, linewidth=1.5, label='Terrain Elevation')
            ax1.plot(distances, track_elevations, '-', color=color_track, linewidth=2.5, label='Track Elevation')
            ax1.tick_params(axis='y')
            min_elev = min(min(terrain_elevations), min(track_elevations))
            max_elev = max(max(terrain_elevations), max(track_elevations))
            avg_elev = np.mean(track_elevations)
            ax2 = ax1.twinx()
            color_gradient = 'blue'
            ax2.set_ylabel('Gradient (m/m)', color=color_gradient)
            # Extend the last gradient to the endpoint for plotting
            gradient_distances = list(distances[:-1]) + [distances[-1]]
            gradient_values = list(gradients) + [gradients[-1]]

            gradient_fill = ax2.fill_between(
                gradient_distances, 
                gradient_values, 
                0,
                where=[g > 0 for g in gradient_values], 
                color='red', 
                alpha=0.3, 
                interpolate=True,
                label='Uphill'
            )
            gradient_fill_neg = ax2.fill_between(
                gradient_distances, 
                gradient_values, 
                0,
                where=[g <= 0 for g in gradient_values], 
                color='blue', 
                alpha=0.3, 
                interpolate=True,
                label='Downhill'
            )
            line2 = ax2.plot(gradient_distances, gradient_values, '-', color='purple', linewidth=1.5, label='Gradient')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
            ax2.tick_params(axis='y', labelcolor=color_gradient)
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            elev_range = max_elev - min_elev
            elev_buffer = 1.0 if elev_range < 1 else elev_range * 0.1
            ax1.set_ylim(min_elev - elev_buffer, max_elev + elev_buffer)
            if gradients:
                max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
                gradient_buffer = max_abs_gradient * 0.1
                ax2.set_ylim(-max_abs_gradient - gradient_buffer, max_abs_gradient + gradient_buffer)
            max_gradient = max(abs(g) for g in gradients) if gradients else 0
            avg_gradient = sum(abs(g) for g in gradients)/len(gradients) if gradients else 0
            stats_text = (
                f"Path Length: {distances[-1]:.2f} km\n"
                f"Max Gradient: {max_gradient:.1%}\n"
                f"Avg Gradient: {avg_gradient:.1%}\n"
                f"Min Terrain Elev: {min(terrain_elevations):.1f} m\n"
                f"Avg Track Elev: {avg_elev:.1f} m\n"
                f"Max Track Elev: {max(track_elevations):.1f} m"
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            lines = [ax1.get_lines()[0], ax1.get_lines()[1], line2[0], gradient_fill, gradient_fill_neg]
            labels = ['Terrain Elevation', 'Track Elevation', 'Gradient', 'Uphill', 'Downhill']
            fig.legend(lines, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
            plt.subplots_adjust(bottom=0.15)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            import os
            os.makedirs('plots', exist_ok=True)
            config_name = title.replace("Railway Path ", "").replace(" Profile", "")
            config_name = config_name.replace(" ", "_").lower()
            filename = f'plots/combined_profile_{config_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved combined profile to {filename}")
            plt.show()
    return (
        GRID_SIZE,
        LineCollection,
        LinearSegmentedColormap,
        MAX_ELEVATION_M,
        MIN_ELEVATION_M,
        RailwayPathOptimizer,
        TERRAIN_SIZE_KM,
        ca,
        generate_terrain,
        np,
        plt,
    )


@app.cell
def _(RailwayPathOptimizer, TERRAIN_SIZE_KM, generate_terrain, np):
    # Generate terrain using the terrain_gen module
    terrain_size = 100
    print("Generating terrain...")

    # Generate base terrain (values in range 0.0-1.0)
    terrain = generate_terrain(
        size=terrain_size,  # Use mostly default settings
        seed=42,            # Random seed for reproducibility
    )

    # Scale terrain to a more realistic elevation range (50-500 meters)
    base_elevation = 50.0  # minimum elevation in meters
    elevation_range = 450.0  # maximum elevation variation in meters
    terrain = base_elevation + (terrain * elevation_range)  # This gives us elevations from 50m to 500m

    print(f"Terrain elevation range: {np.min(terrain):.1f}m to {np.max(terrain):.1f}m")

    # Create optimizer with the generated terrain
    optimizer = RailwayPathOptimizer(terrain=terrain)

    # Create a terrain cost function that penalizes height
    terrain_cost = optimizer.create_terrain_cost_from_array(terrain, 
                                                          terrain_size=(TERRAIN_SIZE_KM, TERRAIN_SIZE_KM))
    optimizer.terrain_cost = terrain_cost

    # Define start and end points (normalized coordinates)
    start = (0.2, 0.2)  # normalized
    end = (0.8, 0.8)    # normalized

    via_points = []

    # Optimize the path with different weight configurations
    # Using fewer configurations to reduce runtime
    # (curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight)
    weight_configurations = [
        (1.0, 1.0, 1.0, 0.0, 1.0, "No terrain cost"),
        (1.0, 1.0, 1.0, 1.0, 1.0, "Terrain aware"),  # More emphasis on path length to avoid excessive detours
        # (2.0, 2.0, 1.0, 1.0, 0.5, "Low Curvature"),  # Prioritize low curvature
        # (1.0, 1.0, 1.0, 5.0, 0.5, "Low Cost"),  # Prioritize low cost
    ]

    for weights in weight_configurations:
        curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight, label = weights

        print(f"Optimizing path with {label} configuration...")

        # For no terrain cost case, use straight line initialization without perturbation
        if terrain_cost_weight == 0:
            n_points = 20  # Fewer points for straight line case
            x_init = np.linspace(start[0], end[0], n_points)  # Use normalized coordinates directly
            y_init = np.linspace(start[1], end[1], n_points)  # Use normalized coordinates directly
            initial_path = (x_init, y_init)
        else:
            n_points = 40
            initial_path = None

        x_coords, y_coords, z_coords = optimizer.optimize_path(
            start_point=start,  # Already normalized
            end_point=end,      # Already normalized
            via_points=via_points,
            max_curvature=0.3,
            max_gradient=0.15,
            weights=(curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight),
            n_points=n_points,
            initial_path=initial_path
        )

        if x_coords is not None and y_coords is not None and z_coords is not None:
            # Plot the result
            optimizer.plot_path(x_coords, y_coords, terrain, z_coords, 
                                f"Railway Path: {label} configuration")
            # Add the combined profile
            optimizer.plot_combined_profile(x_coords, y_coords, terrain, z_coords,
                                        f"Terrain Elevation and Rail Gradient Profile: {label}")
        else:
            print(f"Failed to optimize path with {label} configuration")
    return (
        curvature_change_weight,
        curvature_weight,
        end,
        gradient_weight,
        initial_path,
        label,
        n_points,
        optimizer,
        start,
        terrain,
        terrain_cost,
        terrain_cost_weight,
        terrain_height_scale,
        terrain_size,
        time_weight,
        via_points,
        weight_configurations,
        weights,
        x_coords,
        x_init,
        y_coords,
        y_init,
        z_coords,
    )


if __name__ == "__main__":
    app.run()
