import os
import re
import numpy as np
from astropy.io import fits

# set directory to current directory

dir_path = os.path.dirname(os.path.abspath(__file__))
print("\n",dir_path,"\n")
# get all fits files in directory

def get_number(f):
    """Extract the numeric part from filenames like 'Cl3_553s.fits'"""
    match = re.search(r'_(\d+)s?\.fits', f)
    return int(match.group(1)) if match else 0

# Only include strictly the original science files matching the suffix pattern
fits_files = [f for f in os.listdir(dir_path) if re.search(r'_(\d+)s?\.fits', f)]
# Sort the files sequentially by their extracted number
fits_files.sort(key=get_number)
print("Sorted files matching:", fits_files,"\n")

# read and append all fits files
image_data_list = []

for f in fits_files:
    file_path = os.path.join(dir_path, f)
    print(file_path,"\n")
    
    with fits.open(file_path) as hdul:
        # Assuming the image data is in the primary HDU
        current_data = hdul[0].data
        print(f"Shape of {f}: {current_data.shape}")
        image_data_list.append(current_data)

# save combined fits file   

if image_data_list:
    # Stack into a 3D array (num_files, height, width)
    combined_data = np.array(image_data_list)
    print(f"\nCombined data shape: {combined_data.shape}")
    
    output_path = os.path.join(dir_path, 'combined.fits')
    
    # Calculate WCS parameters for the 3rd axis
    wavelengths = [get_number(f) for f in fits_files]
    start_wave = wavelengths[0]
    
    # Calculate the step (delta) between slices
    # Assumes uniform spacing; if non-uniform, DS9 defaults to linear approx
    if len(wavelengths) > 1:
        step = wavelengths[1] - wavelengths[0]
    else:
        step = 1

    hdu = fits.PrimaryHDU(combined_data)
   # --- Standard WCS Keywords for the 3rd Axis ---
    hdu.header['CTYPE3'] = 'WAVE'          # Coordinate Type
    hdu.header['CUNIT3'] = 'Angstrom'      # Or 'nm', depending on your data
    hdu.header['CRPIX3'] = 1               # Reference pixel (the first slice)
    hdu.header['CRVAL3'] = start_wave      # Coordinate value at the reference pixel
    hdu.header['CDELT3'] = step            # Increment per slice
    # ---------------------------------------------- 
    # Save the original filename and extracted number to the header
    for i, filename in enumerate(fits_files):
        num = get_number(filename)
        hdu.header[f"SLICE{i+1}"] = (filename, f"Original file for slice {i+1}")
        hdu.header[f"WAVE{i+1}"]  = (num, f"Extracted number for slice {i+1}")
        
    hdul_out = fits.HDUList([hdu])
    hdul_out.writeto(output_path, overwrite=True)
    
    print(f"Combined FITS image saved to {output_path}.")