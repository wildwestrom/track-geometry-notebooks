import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter

def generate_terrain(
    size=128,
    frequency=4.0,
    octaves=4,
    persistence=0.5,
    lacunarity=2.0,
    smoothing=0.0,
    seed=None
):
    # Create a seeded noise generator
    if seed is not None:
        noise_gen = OpenSimplex(seed=seed)
    else:
        noise_gen = OpenSimplex()
    
    noise = np.zeros((size, size))

    # Generate simplex noise
    for y in range(size):
        for x in range(size):
            freq = frequency
            value = 0
            amplitude = 1.0
            
            for i in range(octaves):
                nx = x * freq / size
                ny = y * freq / size
                
                # Use the noise generator directly
                value += amplitude * noise_gen.noise2(nx, ny)
                
                amplitude *= persistence
                freq *= lacunarity
                
            noise[y][x] = value

    # Normalize to ensure range is exactly 0 to 1
    min_val = noise.min()
    max_val = noise.max()
    normalized_noise = (noise - min_val) / (max_val - min_val)

    # Apply smoothing if needed
    if smoothing > 0:
        terrain = gaussian_filter(normalized_noise, sigma=smoothing)
    else:
        terrain = normalized_noise

    return terrain
