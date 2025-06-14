import cv2
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from pathlib import Path


def extract_color_value_mapping(colorbar_img, min_val, max_val):
    """Extracts a mapping from color to elevation value using a horizontal colorbar."""
    width = colorbar_img.shape[1]
    values = np.linspace(min_val, max_val, width)
    mid_row = colorbar_img[colorbar_img.shape[0] // 2].reshape(-1, 3)
    return values, mid_row


def convert_heightmap_to_elevation_array(
    image_path, colorbar_path, min_val, max_val, output_npy=None, show_plot=True
):
    """Converts a heightmap image to a 2D elevation array based on a colorbar scale."""
    image_path = Path(image_path)
    colorbar_path = Path(colorbar_path)

    # Load the colorbar image and extract mapping
    color_bar = cv2.imread(str(colorbar_path))
    color_bar = cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB)
    values, colors = extract_color_value_mapping(color_bar, min_val, max_val)

    # Build KDTree for fast nearest-neighbor color lookup
    tree = KDTree(colors.astype(np.float32))

    # Load the heightmap image
    data_img = cv2.imread(str(image_path))
    data_rgb = cv2.cvtColor(data_img, cv2.COLOR_BGR2RGB)
    h, w, _ = data_rgb.shape
    pixels = data_rgb.reshape(-1, 3)

    # Query elevation for each pixel using color matching
    _, idx = tree.query(pixels.astype(np.float32), k=1)
    value_array = values[idx.flatten()].reshape(h, w)

    # Visualization (optional)
    if show_plot:
        plt.imshow(value_array, cmap="terrain")
        plt.colorbar(label="Elevation (m)")
        plt.title(f"Estimated Elevation Map\n{image_path.name}")
        plt.show()

    # Save elevation array (optional)
    if output_npy:
        np.save(str(output_npy), value_array)

    # Info
    print(f"Min elevation: {np.min(value_array)}")
    print(f"Max elevation: {np.max(value_array)}")

    return value_array


# Example usage
if __name__ == "__main__":
    convert_heightmap_to_elevation_array(
        image_path = "test1.jpg",
        colorbar_path = "Schiaparelli_crater_scale.jpg",
        min_val = -4650,
        max_val = 6050,
        output_npy = "test1_elevation.npy",
        show_plot = True
    )