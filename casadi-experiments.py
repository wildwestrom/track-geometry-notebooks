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
    TERRAIN_SIZE_KM = 10.0  # 10km × 10km area
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
            Optimize a railway path between two points.

            Args:
                start_point: (x, y) tuple of starting coordinates
                end_point: (x, y) tuple of ending coordinates
                via_points: List of (x, y) tuples that the path should pass through
                max_curvature: Maximum allowable curvature
                max_gradient: Maximum allowable gradient
                weights: Tuple of (curvature_weight, curvature_change_weight, gradient_weight, 
                         terrain_cost_weight, time_weight)
                n_points: Number of discretization points
                initial_path: Optional tuple of (x_init, y_init) for initialization

            Returns:
                x_coords, y_coords: Arrays of coordinates defining the optimal path
            """
            # Set up optimization variables
            opti = ca.Opti()

            # Path coordinates
            x = opti.variable(n_points)
            y = opti.variable(n_points)

            # Unpack weights
            curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight = weights

            # Fix start and end points
            opti.subject_to(x[0] == start_point[0])
            opti.subject_to(y[0] == start_point[1])
            opti.subject_to(x[-1] == end_point[0])
            opti.subject_to(y[-1] == end_point[1])

            # Explicit boundary constraints to keep path within terrain
            # Small buffer (0.1%) from the edges to prevent numerical issues
            buffer = 0.001
            for i in range(n_points):
                opti.subject_to(x[i] >= self.x_bounds[0] + buffer)
                opti.subject_to(x[i] <= self.x_bounds[1] - buffer)
                opti.subject_to(y[i] >= self.y_bounds[0] + buffer)
                opti.subject_to(y[i] <= self.y_bounds[1] - buffer)

            # Add path continuity constraints to prevent "teleporting" between points
            # Limit the maximum step size between consecutive points
            max_step = 0.2  # Relaxed from 0.15 to 0.2
            for i in range(n_points-1):
                step_size = ca.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
                opti.subject_to(step_size <= max_step)

                # Encourage more uniform step sizes for a smoother path
                if i < n_points-2:
                    next_step = ca.sqrt((x[i+2]-x[i+1])**2 + (y[i+2]-y[i+1])**2)
                    # Step size shouldn't change too abruptly - relaxed constraint
                    opti.subject_to(ca.fabs(step_size - next_step) <= 0.08)  # Increased from 0.05 to 0.08

            # Calculate arc length with improved numerical stability
            ds = []
            for i in range(n_points-1):
                ds.append(ca.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2))

            # Add penalty for points near the boundary
            boundary_violation = 0
            for i in range(n_points):
                # Quadratic penalty that increases as points get closer to the boundaries
                # Scale the penalty based on the normalized distance from the bounds
                x_dist_from_min = (x[i] - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])
                x_dist_from_max = (self.x_bounds[1] - x[i]) / (self.x_bounds[1] - self.x_bounds[0])
                y_dist_from_min = (y[i] - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0])
                y_dist_from_max = (self.y_bounds[1] - y[i]) / (self.y_bounds[1] - self.y_bounds[0])

                boundary_violation += (1/(x_dist_from_min + 1e-3) + 1/(x_dist_from_max + 1e-3) + 
                                    1/(y_dist_from_min + 1e-3) + 1/(y_dist_from_max + 1e-3))

            # Calculate gradients first as they're needed for both cases
            gradients = []
            elevation_changes = []
            cumulative_elevation_gain = 0
            prev_elevation = None

            for i in range(n_points-1):
                # Get elevations at start and end of segment using interpolation
                elev1 = self.terrain_interpolant(ca.vertcat(x[i], y[i]))
                elev2 = self.terrain_interpolant(ca.vertcat(x[i+1], y[i+1]))
                dz = elev2 - elev1  # meters
                elevation_changes.append(dz)

                # Track cumulative elevation gain (only count uphill segments)
                # Use CasADi's conditional functions for symbolic variables
                cumulative_elevation_gain += ca.fmax(0, dz)  # Only add positive elevation changes

                # Calculate horizontal distance in meters (using normalized coordinates)
                dx = (x[i+1] - x[i])  # normalized distance
                dy = (y[i+1] - y[i])  # normalized distance
                dist = ca.sqrt(dx**2 + dy**2) * self.scale_factors['distance'] * 1000  # convert to meters

                # Calculate gradient (rise over run, in m/m)
                gradient = dz / (dist + 1e-6)
                gradients.append(gradient)

                # Add constraints on maximum gradient
                opti.subject_to(gradient <= max_gradient)
                opti.subject_to(gradient >= -max_gradient)

            # Calculate curvature with improved mathematical model
            curvature = []
            curvature_change = []
            direction_changes = []

            for i in range(1, n_points-1):
                # Calculate vectors between consecutive points
                v1x = x[i] - x[i-1]
                v1y = y[i] - y[i-1]
                v2x = x[i+1] - x[i]
                v2y = y[i+1] - y[i]

                # Calculate angle between vectors
                dot_product = v1x*v2x + v1y*v2y
                v1_norm = ca.sqrt(v1x**2 + v1y**2)
                v2_norm = ca.sqrt(v2x**2 + v2y**2)

                # Curvature as angle deviation from straight line
                cos_angle = dot_product / (v1_norm * v2_norm + 1e-6)

                # Constraint so that cos_angle doesn't go outside [-1, 1]
                opti.subject_to(cos_angle <= 1)
                opti.subject_to(cos_angle >= -1)

                # Use a more numerically stable curvature calculation
                # 0 for straight line, 2 for complete reversal
                k = 1 - cos_angle
                curvature.append(k)

                # Track the direction change in degrees for analysis
                direction_change = ca.acos(cos_angle) * 180 / np.pi  # Convert to degrees
                direction_changes.append(direction_change)

                # Add constraint on maximum curvature
                opti.subject_to(k <= max_curvature)

                if i > 1:
                    prev_k = curvature[-2]
                    # Change in curvature (second derivative of path)
                    dk = (k - prev_k)
                    curvature_change.append(dk)

                    # Limit the change in curvature for smooth transitions - relaxed
                    opti.subject_to(ca.fabs(dk) <= 0.15)  # Increased from 0.1 to 0.15

            # Calculate objective terms with improved formulation
            curvature_obj = curvature_weight * sum(k**2 for k in curvature) / len(curvature)
            curvature_change_obj = curvature_change_weight * sum(dk**2 for dk in curvature_change) / max(1, len(curvature_change))

            # Gradient penalty - penalize both high gradients and changes in gradient
            gradient_obj = gradient_weight * sum(g**2 for g in gradients) / len(gradients)

            # Add gradient change penalty for smoother profile
            gradient_changes = [ca.fabs(gradients[i] - gradients[i-1]) for i in range(1, len(gradients))]
            gradient_change_obj = 0.5 * gradient_weight * sum(gc**2 for gc in gradient_changes) / max(1, len(gradient_changes))

            # Calculate terrain cost - now based on work against gravity and excavation costs
            terrain_costs = []
            for i in range(n_points-1):
                # Get elevations
                elevation1 = self.terrain_interpolant(ca.vertcat(x[i], y[i]))
                elevation2 = self.terrain_interpolant(ca.vertcat(x[i+1], y[i+1]))

                # Horizontal distance
                dx = (x[i+1] - x[i])  # normalized distance
                dy = (y[i+1] - y[i])  # normalized distance
                dist = ca.sqrt(dx**2 + dy**2) * self.scale_factors['distance'] # in km

                # Work against gravity (elevation change × distance)
                elev_change = elevation2 - elevation1

                # Different cost for uphill vs downhill
                # Climbing (positive elev_change) is more expensive than descending
                uphill_cost = 3.0 * ca.fmax(0, elev_change) * dist
                downhill_cost = 1.0 * ca.fmax(0, -elev_change) * dist
                work_cost = uphill_cost + downhill_cost

                # Excavation cost - higher elevations require more excavation
                avg_elevation = (elevation1 + elevation2) / 2
                excavation_cost = 0.5 * avg_elevation * dist

                # Total segment cost
                terrain_costs.append(work_cost + excavation_cost)

            cost_obj = terrain_cost_weight * sum(terrain_costs) / len(terrain_costs)

            # Time/distance objective - penalize total path length with higher weight
            path_length = sum(ds) * self.scale_factors['distance']  # in km
            time_obj = time_weight * path_length

            # Add a penalty for excessive total elevation gain
            elevation_gain_obj = 0.2 * terrain_cost_weight * cumulative_elevation_gain / self.scale_factors['distance']

            # Combine all objectives with balanced weights
            boundary_weight = 0.1  # Weight for boundary violation penalty
            objective = (curvature_obj + curvature_change_obj + 
                       gradient_obj + gradient_change_obj + 
                       cost_obj + elevation_gain_obj + 
                       time_obj + boundary_weight * boundary_violation)

            opti.minimize(objective)

            # Set initial guess using a smarter initialization strategy
            if initial_path is not None:
                x_init, y_init = initial_path
            else:
                # Create an initial guess with a slightly curved path that
                # follows lower elevation areas between start and end
                x_init = np.zeros(n_points)
                y_init = np.zeros(n_points)

                # First try a straight line to see elevation profile
                straight_x = np.linspace(start_point[0], end_point[0], n_points)
                straight_y = np.linspace(start_point[1], end_point[1], n_points)

                # Sample terrain along the straight line
                elevations = np.zeros(n_points)
                for i in range(n_points):
                    x_coord = min(max(straight_x[i], 0), 1)
                    y_coord = min(max(straight_y[i], 0), 1)
                    if self.terrain is not None:
                        # Convert normalized coordinates to array indices
                        i_x = int(x_coord * (self.width - 1))
                        i_y = int(y_coord * (self.height - 1))
                        elevations[i] = self.terrain[i_y, i_x]
                    else:
                        elevations[i] = 0

                # Calculate a detour parameter based on terrain variation 
                # (more detour for more varied terrain)
                elevation_range = np.max(elevations) - np.min(elevations)
                relative_range = elevation_range / (np.mean(elevations) + 1e-6)
                detour_factor = min(0.2, relative_range * 0.1)  # Cap at 0.2

                # Create a slight arc to avoid high points
                for i in range(n_points):
                    t = i / (n_points - 1)

                    # Base path is straight line from start to end
                    x_base = (1 - t) * start_point[0] + t * end_point[0]
                    y_base = (1 - t) * start_point[1] + t * end_point[1]

                    # Add a slight arc perpendicular to the straight line
                    # Maximum deviation at the midpoint, zero at endpoints
                    # Direction is perpendicular to straight line
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]

                    # Calculate perpendicular direction (normalized)
                    dist = np.sqrt(dx**2 + dy**2) + 1e-10
                    perp_x = -dy / dist
                    perp_y = dx / dist

                    # Arc shape: 0 at endpoints, maximum at midpoint
                    arc_factor = 4 * t * (1 - t) * detour_factor

                    # Add the arc offset
                    x_init[i] = x_base + perp_x * arc_factor
                    y_init[i] = y_base + perp_y * arc_factor

                    # Ensure within bounds
                    x_init[i] = np.clip(x_init[i], 
                                      self.x_bounds[0] + 2*buffer, 
                                      self.x_bounds[1] - 2*buffer)
                    y_init[i] = np.clip(y_init[i], 
                                      self.y_bounds[0] + 2*buffer, 
                                      self.y_bounds[1] - 2*buffer)

            opti.set_initial(x, x_init)
            opti.set_initial(y, y_init)

            # Solver options
            options = {
                "print_time": False,
                "ipopt": {
                    "max_iter": 3000,  # Increased from 1000
                    "tol": 1e-4,
                    "acceptable_tol": 1e-3,
                    "mu_strategy": "adaptive",
                    "hessian_approximation": "limited-memory",
                    "limited_memory_max_history": 50,
                    "bound_push": 0.01,
                    "bound_frac": 0.01,
                    "warm_start_init_point": "yes",
                    "print_level": 3,
                    "nlp_scaling_method": "gradient-based",
                    "alpha_for_y": "safer-min-dual-infeas",
                    "recalc_y": "yes",
                    "acceptable_iter": 10,  # Added for faster acceptance of approximate solutions
                    "acceptable_obj_change_tol": 1e-3,  # More relaxed convergence check
                    "constr_viol_tol": 1e-4  # Slightly relaxed constraint violation tolerance
                }
            }
            opti.solver('ipopt', options)

            try:
                sol = opti.solve()
                x_coords = sol.value(x)
                y_coords = sol.value(y)
                return x_coords, y_coords
            except Exception as e:
                print(f"Optimization failed: {e}")
                try:
                    x_coords = opti.debug.value(x)
                    y_coords = opti.debug.value(y)
                    print("Returning partial solution from the latest solver iteration.")
                    return x_coords, y_coords
                except:
                    return None, None

        def plot_path(self, x_coords, y_coords, terrain=None, title="Optimized Railway Path"):
            """
            Plot the optimized path with real-world units.
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
                cbar = plt.colorbar(terrain_plot, ax=ax, label='Elevation (meters above sea level)')

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

            # Calculate gradient for coloring
            gradients = []
            segments = []
            for i in range(len(x_km)-1):
                dx = x_km[i+1] - x_km[i]  # km
                dy = y_km[i+1] - y_km[i]  # km

                # Get elevation change in meters
                if terrain is not None:
                    height, width = terrain.shape
                    x_ratio = width / self.scale_factors['distance']
                    y_ratio = height / self.scale_factors['distance']

                    # Get elevations at start and end of segment
                    x1_idx = int(min(max(x_coords[i] * x_ratio, 0), width-1))
                    y1_idx = int(min(max(y_coords[i] * y_ratio, 0), height-1))
                    x2_idx = int(min(max(x_coords[i+1] * x_ratio, 0), width-1))
                    y2_idx = int(min(max(y_coords[i+1] * y_ratio, 0), height-1))

                    elev1 = terrain[y1_idx, x1_idx]  # meters
                    elev2 = terrain[y2_idx, x2_idx]  # meters
                    dz = elev2 - elev1  # meters
                else:
                    dz = 0

                # Calculate true gradient (elevation change / distance)
                # Convert km to m for gradient calculation
                distance_m = np.sqrt(dx**2 + dy**2) * 1000  # convert km to m
                gradient = dz / (distance_m + 1e-6)  # m/m (dimensionless)

                gradients.append(gradient)
                segments.append([(x_km[i], y_km[i]), (x_km[i+1], y_km[i+1])])

            if segments:
                max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
                norm = plt.Normalize(-max_abs_gradient, max_abs_gradient)

                colors = [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.8, 0.0, 0.0)]  # Red -> White -> Red
                n_bins = 256
                cmap = LinearSegmentedColormap.from_list("gradient", colors, N=n_bins)

                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
                lc.set_array(np.array(gradients))
                ax.add_collection(lc)

            # Add white background to legend and stats for better visibility
            ax.legend(loc='lower right', framealpha=1, facecolor='white', edgecolor='black')

            # Calculate statistics
            # Use normalized coordinates for distance calculation
            total_length_km = sum(np.sqrt((x_coords[i+1]-x_coords[i])**2 + 
                                        (y_coords[i+1]-y_coords[i])**2)
                                for i in range(len(x_coords)-1)) * self.scale_factors['distance']

            max_gradient = max(abs(g) for g in gradients) if gradients else 0
            avg_gradient = sum(abs(g) for g in gradients)/len(gradients) if gradients else 0

            if terrain is not None:
                terrain_costs = []
                for i in range(len(x_coords)):
                    height, width = terrain.shape
                    x_ratio = width / self.scale_factors['distance']
                    y_ratio = height / self.scale_factors['distance']
                    x_idx = int(min(max(x_coords[i] * x_ratio, 0), width-1))
                    y_idx = int(min(max(y_coords[i] * y_ratio, 0), height-1))
                    terrain_costs.append(terrain[y_idx, x_idx])

                avg_elev = sum(terrain_costs) / len(terrain_costs)
                max_elev = max(terrain_costs)
                min_elev = min(terrain_costs)

                stats_text = (
                    f"Path Length: {total_length_km:.2f} km\n"
                    f"Max Gradient: {max_gradient:.1%}\n"
                    f"Avg Gradient: {avg_gradient:.1%}\n"
                    f"Min Elevation: {min_elev:.0f} m\n"
                    f"Avg Elevation: {avg_elev:.0f} m\n"
                    f"Max Elevation: {max_elev:.0f} m"
                )
            else:
                stats_text = (
                    f"Path Length: {total_length_km:.2f} km\n"
                    f"Max Gradient: {max_gradient:.1%}\n"
                    f"Avg Gradient: {avg_gradient:.1%}"
                )

            # Add text box with statistics
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

            # Save the plot to a file
            import os
            import datetime
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            # Clean the title to make it filesystem-friendly
            config_name = title.replace("Railway Path with ", "").replace(" Configuration", "")
            config_name = config_name.replace(" ", "_").lower()
            filename = f'plots/railway_path_{config_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filename}")

            plt.show()

        def plot_gradient_profile(self, x_coords, y_coords, terrain=None, title="Railway Path Gradient Profile"):
            """
            Plot a separate scatter plot of path gradient with points spaced by Euclidean distance.
            """
            if terrain is None:
                print("Cannot plot gradient profile without terrain data")
                return

            # Convert coordinates to kilometers
            x_km = x_coords * self.scale_factors['distance']
            y_km = y_coords * self.scale_factors['distance']

            # Calculate cumulative distance along the path
            distances = [0]  # Start with 0
            gradients = []
            elevations = []

            for i in range(len(x_km)-1):
                # Calculate horizontal distance in km
                dx = x_km[i+1] - x_km[i]
                dy = y_km[i+1] - y_km[i]
                segment_distance = np.sqrt(dx**2 + dy**2)  # Euclidean distance in km

                # Skip segments with extremely small distances to avoid division by near-zero
                if segment_distance < 1e-6:
                    continue

                # Add to cumulative distance
                distances.append(distances[-1] + segment_distance)

                # Get elevations at start and end of segment
                height, width = terrain.shape
                x_ratio = width / self.scale_factors['distance']
                y_ratio = height / self.scale_factors['distance']

                x1_idx = int(min(max(x_coords[i] * x_ratio, 0), width-1))
                y1_idx = int(min(max(y_coords[i] * y_ratio, 0), height-1))
                x2_idx = int(min(max(x_coords[i+1] * x_ratio, 0), width-1))
                y2_idx = int(min(max(y_coords[i+1] * y_ratio, 0), height-1))

                elev1 = terrain[y1_idx, x1_idx]  # meters
                elev2 = terrain[y2_idx, x2_idx]  # meters
                elevations.append(elev1)

                dz = elev2 - elev1  # meters

                # Calculate gradient (elevation change / distance)
                distance_m = segment_distance * 1000  # convert km to m
                gradient = dz / distance_m  # m/m (dimensionless)
                gradients.append(gradient)

            # Add the final elevation point
            if len(x_km) > 0:
                height, width = terrain.shape
                x_ratio = width / self.scale_factors['distance']
                y_ratio = height / self.scale_factors['distance']
                x_idx = int(min(max(x_coords[-1] * x_ratio, 0), width-1))
                y_idx = int(min(max(y_coords[-1] * y_ratio, 0), height-1))
                elevations.append(terrain[y_idx, x_idx])

            # Filter out any anomalous gradients (more than 3 standard deviations from mean)
            if gradients:
                gradient_array = np.array(gradients)
                mean_gradient = np.mean(gradient_array)
                std_gradient = np.std(gradient_array)
                threshold = 3 * std_gradient

                # Replace anomalous values with interpolated values from neighbors
                for i in range(len(gradients)):
                    if abs(gradients[i] - mean_gradient) > threshold:
                        # Find previous and next valid gradient
                        prev_valid = next_valid = None
                        prev_idx = i - 1
                        next_idx = i + 1

                        # Look for previous valid gradient
                        while prev_idx >= 0:
                            if abs(gradients[prev_idx] - mean_gradient) <= threshold:
                                prev_valid = gradients[prev_idx]
                                break
                            prev_idx -= 1

                        # Look for next valid gradient
                        while next_idx < len(gradients):
                            if abs(gradients[next_idx] - mean_gradient) <= threshold:
                                next_valid = gradients[next_idx]
                                break
                            next_idx += 1

                        # Interpolate or use nearest valid gradient
                        if prev_valid is not None and next_valid is not None:
                            # Linear interpolation
                            gradients[i] = (prev_valid + next_valid) / 2
                        elif prev_valid is not None:
                            gradients[i] = prev_valid
                        elif next_valid is not None:
                            gradients[i] = next_valid
                        else:
                            # If no valid neighbors, use mean
                            gradients[i] = mean_gradient

            # Create figure for gradient profile
            fig, ax = plt.subplots(figsize=(10, 5))

            # Create scatter plot with gradient color-coding
            if gradients:
                max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
                norm = plt.Normalize(-max_abs_gradient, max_abs_gradient)

                # Use a more intuitive colormap: red for uphill, blue for downhill
                cmap = plt.cm.RdBu_r

                # Plot gradient points with color based on value
                scatter = ax.scatter(distances[1:], gradients, c=gradients, cmap=cmap, 
                                   norm=norm, s=50, edgecolor='black', linewidth=0.5)

                # Add a colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Gradient (m/m)')
                cbar.ax.set_yticklabels([f'{x:.1%}' for x in cbar.ax.get_yticks()])

                # Connect points with line to show progression along path
                ax.plot(distances[1:], gradients, '-', color='gray', alpha=0.5, linewidth=1)

            # Add a zero line
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

            # Calculate statistics (after filtering anomalies)
            if gradients:
                max_gradient = max(abs(g) for g in gradients)
                avg_gradient = sum(abs(g) for g in gradients)/len(gradients)

                # Add text box with statistics
                stats_text = (
                    f"Path Length: {distances[-1]:.2f} km\n"
                    f"Max Gradient: {max_gradient:.1%}\n"
                    f"Avg Gradient: {avg_gradient:.1%}"
                )

                props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

            # Add axis labels and title
            ax.set_xlabel('Distance along path (km)')
            ax.set_ylabel('Gradient (m/m)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

            # Set y-axis limits with a bit of padding
            if gradients:
                padding = max_abs_gradient * 0.1
                ax.set_ylim(-max_abs_gradient - padding, max_abs_gradient + padding)

            plt.tight_layout()

            # Save the figure
            import os
            os.makedirs('plots', exist_ok=True)
            config_name = title.replace("Railway Path ", "").replace(" Gradient Profile", "")
            config_name = config_name.replace(" ", "_").lower()
            filename = f'plots/gradient_profile_{config_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved gradient profile to {filename}")

            # Optional: Create an elevation profile plot too
            if elevations:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(distances, elevations, '-o', color='green', linewidth=2)
                ax.set_xlabel('Distance along path (km)')
                ax.set_ylabel('Elevation (m)')
                ax.set_title(f"Railway Path {title.split(' Gradient Profile')[0]} Elevation Profile")
                ax.grid(True, alpha=0.3)

                # Add statistics
                min_elev = min(elevations)
                max_elev = max(elevations)
                avg_elev = sum(elevations) / len(elevations)

                elev_stats = (
                    f"Min Elevation: {min_elev:.1f} m\n"
                    f"Avg Elevation: {avg_elev:.1f} m\n"
                    f"Max Elevation: {max_elev:.1f} m"
                )

                ax.text(0.05, 0.95, elev_stats, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

                plt.tight_layout()

                # Save elevation profile
                elevation_filename = f'plots/elevation_profile_{config_name}.png'
                plt.savefig(elevation_filename, dpi=300, bbox_inches='tight')
                print(f"Saved elevation profile to {elevation_filename}")

            plt.show()

        def plot_combined_profile(self, x_coords, y_coords, terrain=None, title="Railway Path Profile"):
            """
            Create a combined plot showing both gradient and elevation profiles using dual y-axes.
            """
            if terrain is None:
                print("Cannot plot combined profile without terrain data")
                return
            
            # Convert coordinates to kilometers
            x_km = x_coords * self.scale_factors['distance']
            y_km = y_coords * self.scale_factors['distance']
            
            # Calculate cumulative distance along the path
            distances = [0]  # Start with 0
            gradients = []
            elevations = []
            
            for i in range(len(x_km)-1):
                # Calculate horizontal distance in km
                dx = x_km[i+1] - x_km[i]
                dy = y_km[i+1] - y_km[i]
                segment_distance = np.sqrt(dx**2 + dy**2)  # Euclidean distance in km
                
                # Skip segments with extremely small distances to avoid division by near-zero
                if segment_distance < 1e-6:
                    continue
                    
                # Add to cumulative distance
                distances.append(distances[-1] + segment_distance)
                
                # Get elevations at start and end of segment
                height, width = terrain.shape
                x_ratio = width / self.scale_factors['distance']
                y_ratio = height / self.scale_factors['distance']
                
                x1_idx = int(min(max(x_coords[i] * x_ratio, 0), width-1))
                y1_idx = int(min(max(y_coords[i] * y_ratio, 0), height-1))
                x2_idx = int(min(max(x_coords[i+1] * x_ratio, 0), width-1))
                y2_idx = int(min(max(y_coords[i+1] * y_ratio, 0), height-1))
                
                elev1 = terrain[y1_idx, x1_idx]  # meters
                elev2 = terrain[y2_idx, x2_idx]  # meters
                elevations.append(elev1)
                
                dz = elev2 - elev1  # meters
                
                # Calculate gradient (elevation change / distance)
                distance_m = segment_distance * 1000  # convert km to m
                gradient = dz / distance_m  # m/m (dimensionless)
                gradients.append(gradient)
            
            # Add the final elevation point
            if len(x_km) > 0:
                height, width = terrain.shape
                x_ratio = width / self.scale_factors['distance']
                y_ratio = height / self.scale_factors['distance']
                x_idx = int(min(max(x_coords[-1] * x_ratio, 0), width-1))
                y_idx = int(min(max(y_coords[-1] * y_ratio, 0), height-1))
                elevations.append(terrain[y_idx, x_idx])
                
            # Filter out any anomalous gradients (more than 3 standard deviations from mean)
            if gradients:
                gradient_array = np.array(gradients)
                mean_gradient = np.mean(gradient_array)
                std_gradient = np.std(gradient_array)
                threshold = 3 * std_gradient
                
                # Replace anomalous values with interpolated values from neighbors
                for i in range(len(gradients)):
                    if abs(gradients[i] - mean_gradient) > threshold:
                        # Find previous and next valid gradient
                        prev_valid = next_valid = None
                        prev_idx = i - 1
                        next_idx = i + 1
                        
                        # Look for previous valid gradient
                        while prev_idx >= 0:
                            if abs(gradients[prev_idx] - mean_gradient) <= threshold:
                                prev_valid = gradients[prev_idx]
                                break
                            prev_idx -= 1
                        
                        # Look for next valid gradient
                        while next_idx < len(gradients):
                            if abs(gradients[next_idx] - mean_gradient) <= threshold:
                                next_valid = gradients[next_idx]
                                break
                            next_idx += 1
                        
                        # Interpolate or use nearest valid gradient
                        if prev_valid is not None and next_valid is not None:
                            # Linear interpolation
                            gradients[i] = (prev_valid + next_valid) / 2
                        elif prev_valid is not None:
                            gradients[i] = prev_valid
                        elif next_valid is not None:
                            gradients[i] = next_valid
                        else:
                            # If no valid neighbors, use mean
                            gradients[i] = mean_gradient
            
            # Create the combined figure with two y-axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Elevation profile on primary y-axis (left)
            color_elevation = 'green'
            ax1.set_xlabel('Distance along path (km)')
            ax1.set_ylabel('Elevation (m)', color=color_elevation)
            line1 = ax1.plot(distances, elevations, '-', color=color_elevation, linewidth=2.5, label='Elevation')
            ax1.tick_params(axis='y', labelcolor=color_elevation)
            
            # Calculate elevation statistics
            min_elev = min(elevations)
            max_elev = max(elevations)
            avg_elev = sum(elevations) / len(elevations)
            
            # Create secondary y-axis (right) for gradient
            ax2 = ax1.twinx()
            color_gradient = 'blue'
            ax2.set_ylabel('Gradient (m/m)', color=color_gradient)
            
            # Create a properly aligned distances array for gradients that includes the starting point
            # This fixes the off-by-one error in the gradient plot
            gradient_distances = [distances[i] for i in range(len(distances)-1)]
            
            # Plot gradient as a filled area for better visibility
            gradient_fill = ax2.fill_between(
                gradient_distances, 
                gradients, 
                0,
                where=[g > 0 for g in gradients], 
                color='red', 
                alpha=0.3, 
                interpolate=True,
                label='Uphill'
            )
            
            gradient_fill_neg = ax2.fill_between(
                gradient_distances, 
                gradients, 
                0,
                where=[g <= 0 for g in gradients], 
                color='blue', 
                alpha=0.3, 
                interpolate=True,
                label='Downhill'
            )
            
            # Add the gradient line on top
            line2 = ax2.plot(gradient_distances, gradients, '-', color='purple', linewidth=1.5, label='Gradient')
            
            # Format y-axis as percentage
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
            ax2.tick_params(axis='y', labelcolor=color_gradient)
            
            # Add a zero line for gradient
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            
            # Set appropriate y-limits
            # Set a small buffer for the elevation axis
            elev_range = max_elev - min_elev
            if elev_range < 1:  # If almost flat
                elev_buffer = 1.0
            else:
                elev_buffer = elev_range * 0.1
            ax1.set_ylim(min_elev - elev_buffer, max_elev + elev_buffer)
            
            # Set reasonable limits for gradient axis
            if gradients:
                max_abs_gradient = max(abs(min(gradients)), abs(max(gradients)))
                gradient_buffer = max_abs_gradient * 0.1
                ax2.set_ylim(-max_abs_gradient - gradient_buffer, max_abs_gradient + gradient_buffer)
            
            # Calculate gradient statistics
            max_gradient = max(abs(g) for g in gradients) if gradients else 0
            avg_gradient = sum(abs(g) for g in gradients)/len(gradients) if gradients else 0
            
            # Add combined statistics textbox
            stats_text = (
                f"Path Length: {distances[-1]:.2f} km\n"
                f"Max Gradient: {max_gradient:.1%}\n"
                f"Avg Gradient: {avg_gradient:.1%}\n"
                f"Min Elevation: {min_elev:.1f} m\n"
                f"Avg Elevation: {avg_elev:.1f} m\n"
                f"Max Elevation: {max_elev:.1f} m"
            )
            
            # Add text box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            handles = lines + [gradient_fill, gradient_fill_neg]
            labels = ['Elevation', 'Gradient', 'Uphill', 'Downhill']
            fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))
            
            # Adjust the layout to make room for the legend
            plt.subplots_adjust(bottom=0.15)
            
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Save the combined figure
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

    # Scale terrain to desired height range (0-10 units)
    terrain_height_scale = 10.0
    terrain = terrain * terrain_height_scale

    # Create optimizer with the generated terrain
    optimizer = RailwayPathOptimizer(terrain=terrain)

    # Create a terrain cost function that penalizes height
    terrain_cost = optimizer.create_terrain_cost_from_array(terrain, 
                                                          terrain_size=(TERRAIN_SIZE_KM, TERRAIN_SIZE_KM))
    optimizer.terrain_cost = terrain_cost

    # Define start and end points (in km)
    start = (1 * TERRAIN_SIZE_KM/10, 1 * TERRAIN_SIZE_KM/10)  # Convert to km scale
    end = (9 * TERRAIN_SIZE_KM/10, 9 * TERRAIN_SIZE_KM/10)    # Convert to km scale

    via_points = []

    # Optimize the path with different weight configurations
    # Using fewer configurations to reduce runtime
    # (curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight)
    weight_configurations = [
        (1.0, 1.0, 5.0, 0.0, 1.0, "No terrain cost"),  # Focus on gradients and smooth path
        # (1.0, 1.0, 5.0, 1.0, 2.0, "Terrain aware"),  # More emphasis on path length to avoid excessive detours
        # (5.0, 2.0, 1.0, 1.0, 0.5, "Low Curvature"),  # Prioritize low curvature
        # (1.0, 1.0, 5.0, 5.0, 0.5, "Low Cost"),  # Prioritize low cost
    ]

    for weights in weight_configurations:
        curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight, label = weights

        print(f"Optimizing path with {label} configuration...")

        # For no terrain cost case, use straight line initialization without perturbation
        if terrain_cost_weight == 0:
            n_points = 20  # Fewer points for straight line case
            x_init = np.linspace(start[0]/TERRAIN_SIZE_KM, end[0]/TERRAIN_SIZE_KM, n_points)  # Convert back to 0-1 scale
            y_init = np.linspace(start[1]/TERRAIN_SIZE_KM, end[1]/TERRAIN_SIZE_KM, n_points)  # Convert back to 0-1 scale
            initial_path = (x_init, y_init)
        else:
            n_points = 40
            initial_path = None

        x_coords, y_coords = optimizer.optimize_path(
            start_point=(start[0]/TERRAIN_SIZE_KM, start[1]/TERRAIN_SIZE_KM),  # Convert back to 0-1 scale
            end_point=(end[0]/TERRAIN_SIZE_KM, end[1]/TERRAIN_SIZE_KM),    # Convert back to 0-1 scale
            via_points=via_points,
            max_curvature=0.3,
            max_gradient=0.15,
            weights=(curvature_weight, curvature_change_weight, gradient_weight, terrain_cost_weight, time_weight),
            n_points=n_points,
            initial_path=initial_path
        )

        if x_coords is not None and y_coords is not None:
            # Plot the result
            optimizer.plot_path(x_coords, y_coords, terrain, 
                                f"Railway Path: {label} configuration")
            # Add the combined profile
            optimizer.plot_combined_profile(x_coords, y_coords, terrain,
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
    )


if __name__ == "__main__":
    app.run()
