from noise import snoise2
import numpy as np
from scipy.ndimage import gaussian_filter


def generate_point_of_terrain(nx, ny, octaves=1, persistence=0.5, lacunarity=2):
    return snoise2(
        nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity
    )


# Generate simplex noise
def generate_terrain(
    size=128,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    frequency=4.0,
    smoothing=0.0,
):
    noise = np.zeros((size, size))

    # Generate simplex noise using snoise2
    for y in range(size):
        for x in range(size):
            freq = frequency

            # Calculate noise value using snoise2
            nx = x * freq / size
            ny = y * freq / size
            value = generate_point_of_terrain(
                nx, ny, octaves=octaves, persistence=persistence, lacunarity=lacunarity
            )

            # Store the accumulated noise value
            noise[y][x] = value

    # Normalize to ensure range is exactly 0 to 1
    min_val = noise.min()
    max_val = noise.max()
    normalized_noise = (noise - min_val) / (max_val - min_val)

    # Apply max value scaling
    if smoothing > 0:
        terrain = gaussian_filter(normalized_noise, sigma=smoothing)
    else:
        terrain = normalized_noise

    return terrain
