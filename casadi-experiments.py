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
                import casadi as ca
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
                          max_curvature=0.01, max_gradient=0.05, 
                          weights=(1.0, 2.0, 1.0, 0.5), n_points=100):
            """
            Optimize a railway path between two points.
        
            Args:
                start_point: (x, y) tuple of starting coordinates
                end_point: (x, y) tuple of ending coordinates
                via_points: List of (x, y) tuples that the path should pass through
                max_curvature: Maximum allowable curvature
                max_gradient: Maximum allowable gradient
                weights: Tuple of (curvature_weight, curvature_change_weight, 
                                   cost_weight, time_weight)
                n_points: Number of discretization points
            
            Returns:
                x_coords, y_coords: Arrays of coordinates defining the optimal path
            """
            # Set up optimization variables
            opti = ca.Opti()
        
            # Path coordinates
            x = opti.variable(n_points)
            y = opti.variable(n_points)
        
            # Fix start and end points
            opti.subject_to(x[0] == start_point[0])
            opti.subject_to(y[0] == start_point[1])
            opti.subject_to(x[-1] == end_point[0])
            opti.subject_to(y[-1] == end_point[1])
        
            # If via points are provided, add constraints
            if via_points:
                for i, point in enumerate(via_points):
                    # Find the closest index in our discretization
                    idx = int((i + 1) * n_points / (len(via_points) + 1))
                    opti.subject_to(x[idx] == point[0])
                    opti.subject_to(y[idx] == point[1])
        
            # Calculate arc length
            ds = []
            for i in range(n_points-1):
                ds.append(ca.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2))
        
            # Calculate approximations for curvature
            curvature = []
            curvature_change = []
        
            for i in range(1, n_points-1):
                # First derivatives (velocity)
                dx1 = (x[i] - x[i-1]) / (ds[i-1] + 1e-6)
                dy1 = (y[i] - y[i-1]) / (ds[i-1] + 1e-6)
            
                # Second derivatives (acceleration)
                # Using central difference for better accuracy
                if i < n_points-1:
                    dx2 = (x[i+1] - 2*x[i] + x[i-1]) / ((ds[i-1] + 1e-6)**2)
                    dy2 = (y[i+1] - 2*y[i] + y[i-1]) / ((ds[i-1] + 1e-6)**2)
                else:
                    # For the last point, use backward difference
                    dx2 = (x[i] - 2*x[i-1] + x[i-2]) / ((ds[i-2] + 1e-6)**2)
                    dy2 = (y[i] - 2*y[i-1] + y[i-2]) / ((ds[i-2] + 1e-6)**2)
            
                # Approximate curvature
                k = (dx1*dy2 - dy1*dx2) / ((dx1**2 + dy1**2)**(3/2) + 1e-6)
                curvature.append(k)
            
                # For the change in curvature, calculate dk/ds
                if i > 1:
                    prev_k = curvature[-2]
                    dk = (k - prev_k) / (ds[i-1] + 1e-6)
                    curvature_change.append(dk)
        
            # Calculate construction cost based on terrain
            cost = 0
            for i in range(n_points):
                cost += self.terrain_cost(x[i], y[i])
            
            # Estimate travel time (using a smoothed model to avoid conditional statements)
            travel_time = 0
            for i in range(n_points-1):
                # Create a smooth speed model based on curvature
                if i > 0:
                    k_abs = ca.fabs(curvature[i-1])
                    # Smooth model: v = 1 / (1 + 10*|k|)
                    v = 1.0 / (1.0 + 10.0 * k_abs)
                    travel_time += ds[i] / (v + 1e-6)
                else:
                    travel_time += ds[i]  # Assume unit velocity for first segment
                
            # Scale the objective components to improve numerical conditioning
            # Combine objectives
            alpha1, alpha2, alpha3, alpha4 = weights
        
            # Sum of squares of curvature
            curvature_obj = alpha1 * sum(k**2 for k in curvature)
        
            # Sum of squares of curvature change (MVC term)
            curvature_change_obj = alpha2 * sum(dk**2 for dk in curvature_change)
        
            # Construction cost
            cost_obj = alpha3 * cost
        
            # Travel time
            time_obj = alpha4 * travel_time
        
            # Total objective - scale components to similar magnitudes
            objective = curvature_obj + curvature_change_obj + 0.01 * cost_obj + 0.1 * time_obj
            opti.minimize(objective)
        
            # Add constraints on maximum curvature
            for k in curvature:
                opti.subject_to(ca.fabs(k) <= max_curvature)
            
            # Add constraints on maximum gradient
            for i in range(n_points-1):
                gradient = (y[i+1] - y[i]) / (ca.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2) + 1e-6)
                opti.subject_to(ca.fabs(gradient) <= max_gradient)
            
            # Initial guess - linear interpolation between points with slight perturbation
            # This helps avoid getting stuck in local minima
            x_init = np.zeros(n_points)
            y_init = np.zeros(n_points)
        
            # Create a better initial guess by interpolating through via points
            all_points = [start_point] + (via_points or []) + [end_point]
            for i in range(len(all_points) - 1):
                start_idx = 0 if i == 0 else int(i * n_points / (len(all_points) - 1))
                end_idx = n_points if i == len(all_points) - 2 else int((i + 1) * n_points / (len(all_points) - 1))
            
                # Linear interpolation between these points
                for j in range(start_idx, end_idx):
                    t = (j - start_idx) / max(1, end_idx - start_idx - 1)
                    x_init[j] = (1 - t) * all_points[i][0] + t * all_points[i + 1][0]
                    y_init[j] = (1 - t) * all_points[i][1] + t * all_points[i + 1][1]
                
                    # Add small random perturbation to avoid straight lines
                    if j > 0 and j < n_points - 1:
                        x_init[j] += np.random.normal(0, 0.1)
                        y_init[j] += np.random.normal(0, 0.1)
        
            opti.set_initial(x, x_init)
            opti.set_initial(y, y_init)
            
            # Set up solver with options for better convergence
            options = {
                "print_time": False,
                "ipopt": {
                    "max_iter": 1000,
                    "tol": 1e-4,  # Looser tolerance
                    "acceptable_tol": 1e-2,  # Even looser acceptable tolerance
                    "mu_strategy": "adaptive",  # Adaptive barrier parameter
                    "hessian_approximation": "limited-memory",  # Use L-BFGS approximation for Hessian
                    "limited_memory_max_history": 20,  # Increase history size for L-BFGS
                    "warm_start_init_point": "yes"  # Use warm starting
                }
            }
            opti.solver('ipopt', options)
        
            # Solve the problem with error handling
            try:
                sol = opti.solve()
                x_coords = sol.value(x)
                y_coords = sol.value(y)
                return x_coords, y_coords
            except Exception as e:
                print(f"Optimization failed: {e}")
            
                # Try to extract partial solution if available
                try:
                    x_coords = opti.debug.value(x)
                    y_coords = opti.debug.value(y)
                    print("Returning partial solution from the latest solver iteration.")
                    return x_coords, y_coords
                except:
                    return None, None
            
        def plot_path(self, x_coords, y_coords, terrain=None, title="Optimized Railway Path"):
            """
            Plot the optimized path.
        
            Args:
                x_coords, y_coords: Arrays of coordinates defining the path
                terrain: Optional 2D array representing terrain elevation or cost
                title: Plot title
            """
            fig, ax = plt.subplots(figsize=(12, 10))
        
            # Plot terrain if provided
            if terrain is not None:
                # Using full terrain extent for better visualization
                extent = (0, 10, 0, 10)  # Assuming the terrain is on a 10x10 grid
            
                # Use custom colormap for terrain with better elevation visualization
                terrain_cmap = plt.cm.terrain
                terrain_plot = ax.imshow(terrain, extent=extent, origin='lower', 
                                       alpha=0.8, cmap=terrain_cmap)
                cbar = plt.colorbar(terrain_plot, ax=ax, label='Terrain Height/Cost')
            
                # Overlay contour lines for better elevation visibility
                contour_levels = np.linspace(np.min(terrain), np.max(terrain), 10)
                contour = ax.contour(
                    np.linspace(0, 10, terrain.shape[1]),
                    np.linspace(0, 10, terrain.shape[0]),
                    terrain,
                    levels=contour_levels,
                    colors='black',
                    alpha=0.3,
                    linewidths=0.5
                )
            
                # Add contour labels
                plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
            # Calculate curvature for coloring
            curvature = []
            for i in range(1, len(x_coords)-1):
                # Approximate curvature using finite differences
                dx1 = x_coords[i] - x_coords[i-1]
                dy1 = y_coords[i] - y_coords[i-1]
                dx2 = x_coords[i+1] - x_coords[i]
                dy2 = y_coords[i+1] - y_coords[i]
            
                # Approximate curvature using the formula for discrete curves
                cross_prod = dx1 * dy2 - dx2 * dy1
                norm1 = np.sqrt(dx1**2 + dy1**2)
                norm2 = np.sqrt(dx2**2 + dy2**2)
                k = 2 * cross_prod / (norm1 * norm2 * (norm1 + norm2))
                curvature.append(k)
        
            # Create a colorful line collection based on curvature
            points = np.array([x_coords[1:-1], y_coords[1:-1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
            # Handle empty curvature list (if optimization failed)
            if not curvature:
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Path')
            else:
                # Normalize curvature for coloring
                norm = plt.Normalize(min(curvature), max(curvature))
            
                # Create a colormap that goes from green (low curvature) to red (high curvature)
                cmap = LinearSegmentedColormap.from_list("curvature", ["green", "yellow", "red"])
            
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
                lc.set_array(np.array(curvature))
                ax.add_collection(lc)
            
                # Add a color bar for curvature
                fig.colorbar(lc, ax=ax, label='Curvature')
        
            # Plot start, end, and via points
            ax.plot(x_coords[0], y_coords[0], 'bo', markersize=10, label='Start')
            ax.plot(x_coords[-1], y_coords[-1], 'go', markersize=10, label='End')
        
            # Check for via points (points that aren't at the start or end)
            for i in range(1, len(x_coords)-1):
                # Check if this point is significantly different from neighbors
                # (simplified way to detect via points)
                prev_dist = np.sqrt((x_coords[i]-x_coords[i-1])**2 + (y_coords[i]-y_coords[i-1])**2)
                next_dist = np.sqrt((x_coords[i+1]-x_coords[i])**2 + (y_coords[i+1]-y_coords[i])**2)
                avg_dist = (prev_dist + next_dist) / 2
            
                if abs(prev_dist - next_dist) > 0.2 * avg_dist:
                    ax.plot(x_coords[i], y_coords[i], 'ro', markersize=8)
        
            # Add the main path for better visibility
            ax.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1)
        
            # Add some statistics to the plot
            total_length = sum(np.sqrt((x_coords[i+1]-x_coords[i])**2 + 
                                      (y_coords[i+1]-y_coords[i])**2) 
                              for i in range(len(x_coords)-1))
        
            max_curv = max(abs(k) for k in curvature) if curvature else 0
            avg_curv = sum(abs(k) for k in curvature)/len(curvature) if curvature else 0
        
            # Calculate terrain cost along the path
            if terrain is not None:
                # Sample terrain along the path
                terrain_costs = []
                for i in range(len(x_coords)):
                    # Convert x,y coordinates to indices in the terrain array
                    height, width = terrain.shape
                    x_ratio = width / 10  # Assuming terrain spans 0-10 in x
                    y_ratio = height / 10  # Assuming terrain spans 0-10 in y
                
                    x_idx = int(min(max(x_coords[i] * x_ratio, 0), width-1))
                    y_idx = int(min(max(y_coords[i] * y_ratio, 0), height-1))
                
                    terrain_costs.append(terrain[y_idx, x_idx])
            
                avg_cost = sum(terrain_costs) / len(terrain_costs)
                max_cost = max(terrain_costs)
            
                stats_text = (f"Path Length: {total_length:.2f}\n"
                            f"Max Curvature: {max_curv:.3f}\n"
                            f"Avg Curvature: {avg_curv:.3f}\n"
                            f"Avg Terrain Cost: {avg_cost:.2f}\n"
                            f"Max Terrain Cost: {max_cost:.2f}")
            else:
                stats_text = (f"Path Length: {total_length:.2f}\n"
                            f"Max Curvature: {max_curv:.3f}\n"
                            f"Avg Curvature: {avg_curv:.3f}")
        
            # Add text box with statistics
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True)
            ax.legend(loc='lower right')
            ax.set_aspect('equal')
        
            # Set axis limits to ensure the entire terrain is visible
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
        
            plt.tight_layout()
            plt.show()
        
        def create_example_terrain_cost(self, width, height, obstacles=None):
            """
            Create a sample terrain cost function with optional obstacles.
        
            Args:
                width, height: Dimensions of the terrain
                obstacles: List of tuples (x, y, radius, cost_factor) defining obstacles
            
            Returns:
                cost_function: A function that returns the cost at a given (x, y)
            """
            if obstacles is None:
                obstacles = []
            
            def cost_function(x, y):
                # Base cost
                cost = 1.0
            
                # Add cost from obstacles - avoiding if/else for CasADi compatibility
                for ox, oy, radius, factor in obstacles:
                    dist = ca.sqrt((x - ox)**2 + (y - oy)**2)
                    # Use smooth approximation instead of if/else
                    cost += factor * (1 - dist/radius) * ca.if_else(dist < radius, 1, 0)
                    
                return cost
            
            return cost_function

    # Example usage
    def run_example_with_generated_terrain():
        """Run an example optimization with procedurally generated terrain"""
        # Generate terrain using the terrain_gen module
        terrain_size = 100
        print("Generating terrain...")
    
        # Generate base terrain (values in range 0.0-1.0)
        terrain = generate_terrain(
            size=terrain_size,
            octaves=6,          # Number of octaves for noise
            persistence=0.5,    # How much each octave contributes
            lacunarity=2.0,     # How frequency increases with each octave
            frequency=4.0,      # Base frequency of the noise
            smoothing=1.0       # Gaussian smoothing sigma
        )
    
        # Scale terrain to desired height range (0-10 units)
        terrain_height_scale = 10.0
        terrain = terrain * terrain_height_scale
    
        # Create optimizer with the generated terrain
        optimizer = RailwayPathOptimizer(terrain=terrain)
    
        # Create a terrain cost function that penalizes height
        terrain_cost = optimizer.create_terrain_cost_from_array(terrain, terrain_size=(10, 10))
        optimizer.terrain_cost = terrain_cost
    
        # Define start and end points
        start = (1, 1)
        end = (9, 9)
    
        via_points = [
            # (3, 7),  # Force the path through this point
            # (6, 4)   # And through this point
        ]
    
        # Set random seed for reproducibility
        np.random.seed(42)
    
        # Optimize the path with different weight configurations
        # Using fewer configurations to reduce runtime
        weight_configurations = [
            (1.0, 2.0, 1.0, 0.5, "Balanced"),  # Balanced
            (5.0, 1.0, 1.0, 0.5, "Low Curvature"),  # Prioritize low curvature
            (1.0, 1.0, 5.0, 0.5, "Low Cost"),  # Prioritize low cost
        ]
    
        for weights in weight_configurations:
            alpha1, alpha2, alpha3, alpha4, label = weights
        
            print(f"Optimizing path with {label} configuration...")
            x_coords, y_coords = optimizer.optimize_path(
                start_point=start,
                end_point=end,
                via_points=via_points,
                max_curvature=0.3,
                max_gradient=0.15,
                weights=(alpha1, alpha2, alpha3, alpha4),
                n_points=40  # Reduced from 50 to improve solve time
            )
        
            if x_coords is not None and y_coords is not None:
                # Plot the result
                optimizer.plot_path(x_coords, y_coords, terrain, 
                                  f"Railway Path with {label} Configuration")
            else:
                print(f"Failed to optimize path with {label} configuration")

    if __name__ == "__main__":
        # Uncomment one of these to run the desired example
        # run_example()  # Run with simple obstacle-based terrain
        run_example_with_generated_terrain()  # Run with procedurally generated terrain
    return (
        LineCollection,
        LinearSegmentedColormap,
        RailwayPathOptimizer,
        ca,
        generate_terrain,
        np,
        plt,
        run_example_with_generated_terrain,
    )


if __name__ == "__main__":
    app.run()
