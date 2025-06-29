import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter


class TerrainGen:
    def __init__(self):
        self.terrain: np.ndarray = np.zeros((32, 32))
        self.size: int = 32
        self.frequency: float = 4.0
        self.octaves: int = 4
        self.persistence: float = 0.5
        self.lacunarity: float = 2.7
        self.smoothing: float = 0.0
        self.baseline: float = 0.2  # Center height
        self.dip_depth: float = 0.1  # How deep valleys can go below baseline
        self.peak_sharpness: float = 1.0  # Exponent for sharp peaks

    def generate_terrain(
        self,
        seed: int,
    ) -> np.ndarray:
        noise_gen = OpenSimplex(seed)
        noise = np.zeros((self.size, self.size))

        # Define a function to calculate noise for an entire row
        def process_row(y):
            row_values = []

            for x in range(self.size):
                freq = self.frequency
                value = 0
                amplitude = 1.0

                for i in range(self.octaves):
                    nx = x * freq / self.size
                    ny = y * freq / self.size
                    value += amplitude * noise_gen.noise2(nx, ny)
                    amplitude *= self.persistence
                    freq *= self.lacunarity

                row_values.append(value)

            return (y, row_values)

        # Process all rows
        results = map(process_row, range(self.size))

        # Assign the results to the noise array
        for y, row_values in results:
            noise[y] = row_values

        # Normalize to ensure range is exactly 0 to 1
        min_val = noise.min()
        max_val = noise.max()
        normalized_noise = (noise - min_val) / (max_val - min_val)

        # Apply baseline, dip, and peak shaping
        # 1. Shift to baseline
        terrain = normalized_noise - 0.5
        # 2. Scale dips and peaks
        terrain = np.where(terrain < 0, terrain * self.dip_depth / 0.5, terrain)
        # 3. Apply peak sharpness (exponentiation for positive values)
        terrain = np.where(terrain > 0, terrain**self.peak_sharpness, terrain)
        # 4. Shift back to baseline
        terrain = terrain + self.baseline
        # 5. Clip to [0, 1]
        terrain = np.clip(terrain, 0, 1)

        # Apply smoothing if needed
        if self.smoothing > 0:
            terrain = gaussian_filter(terrain, sigma=self.smoothing)

        self.terrain = terrain

        return terrain
