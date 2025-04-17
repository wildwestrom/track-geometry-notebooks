import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os
from terrain_gen import generate_terrain
from numpy import ndarray
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

# Define physical constants and scales
TERRAIN_SIZE_KM: float = 50.0  # Length of each side in km
MIN_ELEVATION_M: float = 10.0  # m above sea level
MAX_ELEVATION_M: float = 500.0  # m above sea level
GRID_SIZE: int = 128  # Number of grid points in each dimension
DEFAULT_FIGSIZE: tuple[int, int] = (14, 8)


class RailwayPathOptimizer:
    def __init__(self):
        print("Generating terrain...")
        terrain = MIN_ELEVATION_M + (
            generate_terrain(
                size=GRID_SIZE,
                smoothing=1.0,
                seed=42,
            )
            * MAX_ELEVATION_M
            - MIN_ELEVATION_M
        )
        self.terrain = terrain
        print(
            f"Terrain elevation range: {MIN_ELEVATION_M:.1f}m to {MAX_ELEVATION_M:.1f}m"
        )
        self.height, self.width = self.terrain.shape
        self.x_bounds = (0, 1)
        self.y_bounds = (0, 1)
        self.scale_factors = {
            "distance": TERRAIN_SIZE_KM,
            "elevation": MAX_ELEVATION_M - MIN_ELEVATION_M,
            "min_elevation": MIN_ELEVATION_M,
        }
        self.default_figsize = DEFAULT_FIGSIZE

    def harsh_gradient_penalty(self, gc):
        abs_gc = abs(gc)
        if abs_gc <= 0.04:
            return 10 * gc**2
        else:
            return 10 * 0.04**2 + (np.exp(8 * (abs_gc - 0.04)) - 1)

    def curvature_penalty(self, prev, curr, next_):
        # prev, curr, next_ are (i, j) tuples
        v1 = np.array([curr[0] - prev[0], curr[1] - prev[1]])
        v2 = np.array([next_[0] - curr[0], next_[1] - curr[1]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        curvature = angle / (norm1 + norm2 + 1e-6)
        return curvature**2  # quadratic penalty

    def edge_cost(self, prev, curr, next_, dx, dy):
        # Distance
        dxy = np.sqrt(
            (next_[0] - curr[0]) ** 2 * dx**2 + (next_[1] - curr[1]) ** 2 * dy**2
        )
        # Elevation
        elev_curr = self.terrain[curr[0], curr[1]]
        elev_next = self.terrain[next_[0], next_[1]]
        dz = elev_next - elev_curr
        grade = dz / (dxy + 1e-6)
        grad_penalty = self.harsh_gradient_penalty(grade)
        # Curvature penalty (if prev exists)
        curv_penalty = 0.0
        curvature = 0.0
        if prev is not None:
            curv_penalty = self.curvature_penalty(prev, curr, next_)
            # For speed limit
            v1 = np.array([curr[0] - prev[0], curr[1] - prev[1]])
            v2 = np.array([next_[0] - curr[0], next_[1] - curr[1]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                curvature = angle / (norm1 + norm2 + 1e-6)
        # --- Construction cost (tunnels, bridges, cut/fill) ---
        tunnel_threshold = -10.0  # meters below terrain
        bridge_threshold = 10.0  # meters above terrain
        bridge_tunnel_multiplier = 3.0
        tunnel_cost_per_m = 10000.0 * bridge_tunnel_multiplier
        bridge_cost_per_m = 5000.0 * bridge_tunnel_multiplier
        excavation_cost_per_m3 = 70.0
        embankment_cost_per_m3 = 100.0
        track_width = 10.0  # meters
        offset1 = elev_curr - elev_curr  # always 0
        offset2 = elev_next - elev_next  # always 0
        # For a grid-based model, use dz as the offset for cut/fill
        tunnel1 = max(0, -(dz + tunnel_threshold))
        tunnel2 = max(0, -(dz + tunnel_threshold))
        tunnel_cost = tunnel_cost_per_m * (tunnel1 + tunnel2) / 2 * dxy
        bridge1 = max(0, dz - bridge_threshold)
        bridge2 = max(0, dz - bridge_threshold)
        bridge_cost = bridge_cost_per_m * (bridge1 + bridge2) / 2 * dxy
        cut1 = max(0, -dz)
        cut2 = max(0, -dz)
        cutting_cost = excavation_cost_per_m3 * (cut1 + cut2) / 2 * dxy * track_width
        fill1 = max(0, dz)
        fill2 = max(0, dz)
        embankment_cost = (
            embankment_cost_per_m3 * (fill1 + fill2) / 2 * dxy * track_width
        )
        construction_cost = tunnel_cost + bridge_cost + cutting_cost + embankment_cost
        # --- Speed limits and travel time ---
        max_speed = 80.0  # m/s
        max_accel = 0.7  # m/s^2
        max_lateral_accel = 1.0  # m/s^2
        # Curvature-limited speed
        if curvature > 0:
            v_curve = np.sqrt(max_lateral_accel / (abs(curvature) + 1e-8))
        else:
            v_curve = max_speed
        # Gradient-limited speed (downhill braking)
        if grade < 0:
            v_grade = np.sqrt(2 * max_accel / (abs(grade) + 1e-8))
        else:
            v_grade = max_speed
        v_limit = min(v_curve, v_grade, max_speed)
        # Travel time for this segment
        travel_time = dxy / (v_limit + 1e-6)
        # Total cost: weighted sum (weights can be tuned)
        total_cost = (
            dxy
            + grad_penalty
            + curv_penalty
            + 0.00001 * construction_cost
            + 0.1 * travel_time
        )
        return total_cost

    def optimize_path(self, start_point, end_point, n_points=40):
        height, width = self.terrain.shape
        dx = self.scale_factors["distance"] * 1000 / (width - 1)
        dy = self.scale_factors["distance"] * 1000 / (height - 1)
        N = height * width
        adj = lil_matrix((N, N))
        # For each node, connect to 4 neighbors and compute cost
        for i in range(height):
            for j in range(width):
                idx = i * width + j
                curr = (i, j)
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < height and 0 <= nj < width:
                        next_ = (ni, nj)
                        # For curvature, use previous as None for now (can be improved with A* or by storing direction)
                        cost = self.edge_cost(None, curr, next_, dx, dy)
                        nidx = ni * width + nj
                        adj[idx, nidx] = cost
        # Dijkstra's algorithm
        start_idx = int(start_point[1] * (height - 1)) * width + int(
            start_point[0] * (width - 1)
        )
        end_idx = int(end_point[1] * (height - 1)) * width + int(
            end_point[0] * (width - 1)
        )
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
        # Elevation profile
        z_resampled = np.array(
            [
                float(
                    self.terrain[
                        int(y_resampled[i] * (height - 1)),
                        int(x_resampled[i] * (width - 1)),
                    ]
                )
                for i in range(n_points)
            ]
        )
        return x_resampled, y_resampled, z_resampled

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
            distance_m = np.sqrt(dx**2 + dy**2) * 1000  # convert km to m
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
                np.sqrt(
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
        filename = "plots/railway_path.png"
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
            segment_distance = np.sqrt(dx**2 + dy**2)
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
        filename = "plots/combined_profile.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved combined profile to {filename}")
        plt.show()
