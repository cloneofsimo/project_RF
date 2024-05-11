from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def fixed_grid_hsv_quantization(image_path, h_bins=18, s_bins=4, v_bins=4):
    # Load the image
    image = Image.open(image_path)
    image = image.convert("RGB")

    # Convert RGB to HSV
    image_hsv = image.convert("HSV")
    data_hsv = np.array(image_hsv)
    data_hsv = data_hsv.reshape((-1, 3))

    # Scale Hue from 0-255 to 0-360, Saturation and Value from 0-255 to 0-100
    data_hsv = data_hsv * np.array([360 / 255, 100 / 255, 100 / 255])

    # Calculate the bin indices for each pixel
    h_indices = np.floor(data_hsv[:, 0] / (360 / h_bins)).astype(int)
    s_indices = np.floor(data_hsv[:, 1] / (100 / s_bins)).astype(int)
    v_indices = np.floor(data_hsv[:, 2] / (100 / v_bins)).astype(int)

    # Flatten the indices into a single dimension
    indices = h_indices * s_bins * v_bins + s_indices * v_bins + v_indices

    # Create histogram
    histogram = np.bincount(indices, minlength=h_bins * s_bins * v_bins)
    histogram = histogram / np.sum(histogram)  # Normalize histogram

    # Return the histogram
    return histogram


# Example usage
image_path = "test.png"
histogram = fixed_grid_hsv_quantization(image_path, h_bins=18, s_bins=4, v_bins=4)

# Print the histogram
print("Histogram of Quantized Colors:", histogram)

# Optional: Visualize the histogram
plt.bar(range(len(histogram)), histogram)
plt.title("HSV Color Histogram")
plt.xlabel("Bins")
plt.ylabel("Normalized Frequency")
plt.show()
plt.savefig("histogram.png")


def resize_to_2x2(image_path):
    # Load the image from the specified path
    image = Image.open(image_path)

    # Resize the image to 2x2 using the nearest neighbor algorithm
    resized_image = image.resize((2, 2), Image.NEAREST)

    return resized_image


# Example usage
image_path = "test.png"
resized_image = resize_to_2x2(image_path)

resized_image.save("resized_image.png")
