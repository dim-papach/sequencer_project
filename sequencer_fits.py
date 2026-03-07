import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.ndimage import gaussian_filter
import sequencer

fits_file = "combined_cube.fits"

DPI = 200

print("Loading data...")

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    wcs = WCS(hdul[0].header)
    print("Data loaded.")

output_dir = "original"
os.makedirs(output_dir, exist_ok=True)

n_slices = data.shape[0]
cols = int(np.ceil(np.sqrt(n_slices)))
rows = int(np.ceil(n_slices / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    if i < n_slices:
        ax.imshow(data[i], cmap='viridis', origin='lower')
        ax.set_title(f"Slice {i}")
        plt.imsave(os.path.join(output_dir, f"slice_{i}.png"), data[i], cmap='viridis', origin='lower')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "grid.png"), dpi=DPI)
plt.close()


# find the 2d spectra of the images, and visualize them
# instead of plotting the the (x,y) image, plot the (x,z) image
# where z is the spectral dimension

output_dir = "spectral_projections"
os.makedirs(output_dir, exist_ok=True)

n_lambda, n_y, n_x = data.shape

# --- 1. Spectral-X Slices (Slicing along Y) ---
# We iterate through Y to see the spectrum across the X-axis
rows = int(np.ceil(np.sqrt(n_y)))
cols = int(np.ceil(n_y / rows))

fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
for i in range(n_y):
    ax = axes.flatten()[i]
    # Slice: all lambda, fixed y, all x -> Shape: (n_lambda, n_x)
    spectral_x = data[:, i, :] 
    
    ax.imshow(np.log10(spectral_x + 1e-10), cmap='inferno', origin='lower', aspect='auto')
    ax.set_title(f"Y-row: {i}", fontsize=8)
    ax.axis('off')

# Clean up empty subplots
for j in range(i + 1, len(axes.flatten())):
    axes.flatten()[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectra_x_grid.png"), dpi=DPI)
plt.close()

# --- 2. Spectral-Y Slices (Slicing along X) ---
# We iterate through X to see the spectrum across the Y-axis
rows_x = int(np.ceil(np.sqrt(n_x)))
cols_x = int(np.ceil(n_x / rows_x))

fig, axes = plt.subplots(rows_x, cols_x, figsize=(20, 20))
for i in range(n_x):
    ax = axes.flatten()[i]
    # Slice: all lambda, all y, fixed x -> Shape: (n_lambda, n_y)
    spectral_y = data[:, :, i] 
    
    ax.imshow(np.log10(spectral_y + 1e-10), cmap='inferno', origin='lower', aspect='auto')
    ax.set_title(f"X-col: {i}", fontsize=8)
    ax.axis('off')

for j in range(i + 1, len(axes.flatten())):
    axes.flatten()[j].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectra_y_grid.png"), dpi=DPI)
plt.close()