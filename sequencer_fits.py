import os
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split

import sequencer

fits_file = "combined_cube.fits"

DPI = 200

print("Loading data...")

with fits.open(fits_file) as hdul:
    data = hdul[0].data
    wcs = WCS(hdul[0].header)
    print("Data loaded.")

# output_dir = "original"
# os.makedirs(output_dir, exist_ok=True)

n_slices = data.shape[0]
cols = data .shape[1]
rows = data.shape[2]

print("Number of slices: ", n_slices)
print("Number of rows: ", rows)
print("Number of columns: ", cols)

# fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
# for i, ax in enumerate(axes.flat):
#     if i < n_slices:
#         ax.imshow(data[i], cmap='viridis', origin='lower')
#         ax.set_title(f"Slice {i}")
#         plt.imsave(os.path.join(output_dir, f"slice_{i}.png"), data[i], cmap='viridis', origin='lower')
#     ax.axis('off')

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "grid.png"), dpi=DPI)
# plt.close()


# # find the 2d spectra of the images, and visualize them
# # instead of plotting the the (x,y) image, plot the (x,z) image
# # where z is the spectral dimension

# output_dir = "spectral_projections"
# os.makedirs(output_dir, exist_ok=True)

# n_lambda, n_y, n_x = data.shape

# # --- 1. Spectral-X Slices (Slicing along Y) ---
# # We iterate through Y to see the spectrum across the X-axis
# rows = int(np.ceil(np.sqrt(n_y)))
# cols = int(np.ceil(n_y / rows))

# fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
# for i in range(n_y):
#     ax = axes.flatten()[i]
#     # Slice: all lambda, fixed y, all x -> Shape: (n_lambda, n_x)
#     spectral_x = data[:, i, :] 
    
#     ax.imshow(np.log10(spectral_x + 1e-10), cmap='inferno', origin='lower', aspect='auto')
#     ax.set_title(f"Y-row: {i}", fontsize=8)
#     ax.axis('off')

# # Clean up empty subplots
# for j in range(i + 1, len(axes.flatten())):
#     axes.flatten()[j].axis('off')

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "spectra_x_grid.png"), dpi=DPI)
# plt.close()

# # --- 2. Spectral-Y Slices (Slicing along X) ---
# # We iterate through X to see the spectrum across the Y-axis
# rows_x = int(np.ceil(np.sqrt(n_x)))
# cols_x = int(np.ceil(n_x / rows_x))

# fig, axes = plt.subplots(rows_x, cols_x, figsize=(20, 20))
# for i in range(n_x):
#     ax = axes.flatten()[i]
#     # Slice: all lambda, all y, fixed x -> Shape: (n_lambda, n_y)
#     spectral_y = data[:, :, i] 
    
#     ax.imshow(np.log10(spectral_y + 1e-10), cmap='inferno', origin='lower', aspect='auto')
#     ax.set_title(f"X-col: {i}", fontsize=8)
#     ax.axis('off')

# for j in range(i + 1, len(axes.flatten())):
#     axes.flatten()[j].axis('off')

# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "spectra_y_grid.png"), dpi=DPI)
# plt.close()

# # choose a random spectral slice and plot it


# Sequencer algorithm

# The Sequencer is a manifold learning algorithm that identifies the main trend in a dataset.
# It works by calculating distances between all pairs of data points using multiple metrics 
# and scales, constructing a graph, and then finding the optimal 1D sequence that 
# minimizes the total distance between adjacent points


dummy_spectrum = data[:, 120, :]
# Sequencer expects the objects to be sequence as rows (shape: obj, features)
objects_list = dummy_spectrum.T

# Add an offset to ensure all values are strictly positive, as required by Sequencer
min_val = np.min(objects_list)
if min_val <= 0:
    objects_list = objects_list - min_val + 1e-5

# Create a simple uniform grid for the feature dimension
grid = np.arange(objects_list.shape[1])
# Define the scales to evaluate
sub_scale_list = [1,2,5,10,15]

# Define distance metrics to evaluate
estimator_list = ['EMD', 'energy', 'L2', 'KL']

scale_list = [sub_scale_list] * len(estimator_list)

# Instantiate the Sequencer
seq = sequencer.Sequencer(grid, objects_list, estimator_list, scale_list)

# Define output directory and execute
seq_out_dir = "sequencer_output"
os.makedirs(seq_out_dir, exist_ok=True)

num_of_models = len(estimator_list) * len(sub_scale_list)
num_estimators = int(np.sqrt(num_of_models))    

print("Number of models: ", num_of_models)
print("Number of estimators: ", num_estimators)

final_elongation, final_sequence = seq.execute(seq_out_dir, to_average_N_best_estimators=True, 
                                                number_of_best_estimators=num_estimators,
                                                #to_calculate_distance_matrices=False,
                                                #distance_matrices_inpath= seq_out_dir + "/distance_matrices.pkl",
                                                to_use_parallelization=True,
                                                num_cores=4)

print("Sequencer completed.")
print(f"Final Elongation: {final_elongation}")

# Sort the original array according to the sequence
sequenced_dummy_spectrum = dummy_spectrum[:, final_sequence]

print("\n--- Running Sequencer on the other axis (Lambda) ---")
# For the second axis, objects are the lambda indices, and features are the sorted X indices
objects_list_2 = sequenced_dummy_spectrum

# Ensure strictly positive values
min_val_2 = np.min(objects_list_2)
if min_val_2 <= 0:
    objects_list_2 = objects_list_2 - min_val_2 + 1e-5

grid_2 = np.arange(objects_list_2.shape[1])

# Initialize the Sequencer for the second axis
seq2 = sequencer.Sequencer(grid_2, objects_list_2, estimator_list, scale_list)

seq_out_dir_2 = "sequencer_output_2"
os.makedirs(seq_out_dir_2, exist_ok=True)

# Execute sequencer on the second axis
final_elongation_2, final_sequence_2 = seq2.execute(seq_out_dir_2, to_average_N_best_estimators=True, 
                                                number_of_best_estimators=num_estimators,
                                                to_use_parallelization=True,
                                                num_cores=4)

print("Second Sequencer completed.")
print(f"Final Elongation (Lambda axis): {final_elongation_2}")

# Sort the lambda axis of the already X-sequenced spectrum
double_sequenced_spectrum = sequenced_dummy_spectrum[final_sequence_2, :]



# Plot all three stages
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Original Spectrum
im1 = axes[0].imshow(np.log10(dummy_spectrum + 1e-10), cmap='inferno', origin='lower', aspect='auto')
axes[0].set_title("Original Spectrum")
axes[0].set_xlabel('Object Index (X)')
axes[0].set_ylabel('Wavelength/Lambda Index')
plt.colorbar(im1, ax=axes[0], label='log10(Intensity)')

# Sequenced Spectrum (X axis)
im2 = axes[1].imshow(np.log10(sequenced_dummy_spectrum + 1e-10), cmap='inferno', origin='lower', aspect='auto')
axes[1].set_title(f"Sequenced X (Elongation: {final_elongation:.2f})")
axes[1].set_xlabel('Sorted Object Index (X)')
axes[1].set_ylabel('Wavelength/Lambda Index')
plt.colorbar(im2, ax=axes[1], label='log10(Intensity)')

# Sequenced Spectrum (X and Lambda axes)
im3 = axes[2].imshow(np.log10(double_sequenced_spectrum + 1e-10), cmap='inferno', origin='lower', aspect='auto')
axes[2].set_title(f"Sequenced X & Lambda\n(Elongation: {final_elongation_2:.2f})")
axes[2].set_xlabel('Sorted Object Index (X)')
axes[2].set_ylabel('Sorted Wavelength/Lambda Index')
plt.colorbar(im3, ax=axes[2], label='log10(Intensity)')

plt.tight_layout()
output_path = os.path.join(seq_out_dir_2, "double_sequenced_dummy_spectrum.png")
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Double-sequenced image saved to {output_path}")

# --- GPR Smoothing / Imputation on Final Matrix ---
print("Applying GPR to double_sequenced_spectrum...")
ds_original = double_sequenced_spectrum.copy()

nx, ny = double_sequenced_spectrum.shape
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

coords = np.vstack([X.ravel(), Y.ravel()]).T
target = np.log10(double_sequenced_spectrum + 1e-10).flatten()

mask = ~np.isnan(target)
X_train, x_test, Y_train, y_test = train_test_split(coords[mask], target[mask], test_size=1)

print(f"Fitting GP on {len(X_train)} available pixels...")
kernel = C(1.0) * RBF(length_scale=[0.1, 0.1]) + WhiteKernel(noise_level=1e-5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

gp.fit(X_train, Y_train)

print("Predicting values over the grid...")
y_pred, sigma = gp.predict(coords, return_std=True)

double_sequenced_spectrum = y_pred.reshape(nx, ny)

# Visualization of the reconstruction
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(np.log10(ds_original + 1e-10), aspect='auto', origin='lower', cmap='inferno')
axes[0].set_title("Original Double Sequenced Matrix")

axes[1].imshow(np.log10(double_sequenced_spectrum + 1e-10), aspect='auto', origin='lower', cmap='inferno')
axes[1].set_title("GPR Smoothing / Reconstruction")

plt.tight_layout()
plt.savefig(os.path.join(seq_out_dir_2, "final_gpr_reconstruction.png"), dpi=200)
plt.close()
print("Final GPR visualization saved.")
# ------------------------------------
