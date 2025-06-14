from pathlib import Path
from heightmapToData import convert_heightmap_to_elevation_array

# Paths
heightmap_dir = Path("data/heightmaps")
output_dir = Path("data/elevation")
colorbar_path = "Schiaparelli_crater_scale.jpg"  # change if needed

# Value range of colorbar
min_val, max_val = -4650, 6050

# Ensure output dir exists
output_dir.mkdir(parents=True, exist_ok=True)

# Convert all .jpg and .png files
for img_file in heightmap_dir.glob("*"):
    if img_file.suffix.lower() not in [".jpg", ".png"]:
        continue

    output_file = output_dir / f"{img_file.stem}.npy"

    print(f"Converting: {img_file.name}")
    convert_heightmap_to_elevation_array(
        image_path=img_file,
        colorbar_path=colorbar_path,
        min_val=min_val,
        max_val=max_val,
        output_npy=output_file,
        show_plot=True  # turn to True to preview each one
    )