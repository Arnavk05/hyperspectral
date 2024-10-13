import spectral
import numpy as np
import matplotlib.pyplot as plt


# Load hyperspectral data
input_folder = r"C:\Users\arnav\Desktop\HyperSpectral\20240730181428001001" #folder with all data
base_name = "20240730181428001001_00000000_vnir" #basename for both hdr and hsi files
header_file = f"{input_folder}/{base_name}.hdr"
data_file = f"{input_folder}/{base_name}.hsi"

# Read the hyperspectral image
spectral_image = spectral.io.envi.open(header_file, data_file)
hyperspectral_data = spectral_image.load()

# Print the shape of the data array for analysis
print("Shape of hyperspectral data:", hyperspectral_data.shape)

# NDVI Calculation
red_band_index = 40  # Adjust based on your data - Go to .hdr file and select index of band closest to 630-700nm
nir_band_index = 70   # Adjust based on your data - select index 750nm-2000nm

red = hyperspectral_data[:, :, red_band_index].squeeze()
nir = hyperspectral_data[:, :, nir_band_index].squeeze()
ndvi = (nir - red) / (nir + red + 1e-10)  #Calculation of NDVI and avoid division by 0

# Normalize NDVI for visualization
ndvi_normalized = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())
plt.hist(ndvi.flatten(), bins=50)

#Histogram of NDVI values to visualize any issues in the data
plt.title('Histogram of NDVI Values')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')
plt.show()

# Visualize NDVI
plt.figure(figsize=(10, 5))
plt.imshow(ndvi_normalized, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.title('NDVI Visualization')
plt.axis('off')
plt.show()

# Heat Signature Analysis 
heat_signature_band_index = 70  #Typically choose 8000-14000 nm. This data only goes to 1000 so use ~800nm as a proxy
heat_signature_band = hyperspectral_data[:, :, heat_signature_band_index].squeeze()

# Normalize for visualization
heat_signature_normalized = (heat_signature_band - heat_signature_band.min()) / (heat_signature_band.max() - heat_signature_band.min())

# Visualize Heat Signature
plt.figure(figsize=(10, 5))
plt.imshow(heat_signature_normalized, cmap='hot')
plt.colorbar(label='Heat Signature Intensity')
plt.title('Heat Signature Visualization')
plt.axis('off')
plt.show()

# Extract wavelengths
wavelengths = np.array(spectral_image.bands.centers)

# Define indices for Red, Green, and Blue bands based on your data
# Adjust these indices based on your specific requirements
r_band_index = 45  # Example index for Red (~650 nm)
g_band_index = 40  # Example index for Green (~550 nm)
b_band_index = 30  # Example index for Blue (~450 nm)

# Create an RGB image
rgb_image = np.zeros((hyperspectral_data.shape[0], hyperspectral_data.shape[1], 3))

# Fill the RGB channels
rgb_image[:, :, 0] = hyperspectral_data[:, :, r_band_index].squeeze()  # Red channel
rgb_image[:, :, 1] = hyperspectral_data[:, :, g_band_index].squeeze()  # Green channel
rgb_image[:, :, 2] = hyperspectral_data[:, :, b_band_index].squeeze()  # Blue channel

# Normalize the RGB values to [0, 1]
rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

# Display the RGB image
plt.imshow(rgb_image)
plt.axis('off')  # Turn off axis
plt.title("RGB Visualization")
plt.show()