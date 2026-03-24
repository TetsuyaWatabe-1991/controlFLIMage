import os
from pathlib import Path
from math import ceil

from PIL import Image

# Allow very large images (disable decompression bomb protection)
Image.MAX_IMAGE_PIXELS = None


def make_tiled_png(
    folder: str,
    output_name: str = "tiled_GFP_2d.png",
    n_cols: int = 21,
    resize_to: tuple[int, int] | None = None,
    background_color=(0, 0, 0),
):
    """Make a tiled PNG from all PNG images in a folder.

    Args:
        folder: Folder path that contains PNG images.
        output_name: Output PNG file name (saved in the same folder).
        n_cols: Number of columns per row.
        resize_to: (width, height) to resize all images.
                   If None, use (150, 150) so that each tile has
                   maximum side length of 150 pixels.
        background_color: RGB tuple for background color.
    """
    folder_path = Path(folder)

    # Collect PNG files in filename order
    png_files = sorted(
        [p for p in folder_path.iterdir() if p.suffix.lower() == ".png"],
        key=lambda p: p.name,
    )

    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {folder_path}")

    # Load first image to determine tile size
    first_image = Image.open(png_files[0])

    # Decide tile size
    if resize_to is None:
        # Scale first image so that its longest side becomes 150 pixels
        max_side = max(first_image.width, first_image.height)
        scale = 150 / max_side if max_side > 0 else 1.0
        tile_w = int(first_image.width * scale)
        tile_h = int(first_image.height * scale)
    else:
        tile_w, tile_h = resize_to

    # Load all images (including first one)
    images = [first_image] + [Image.open(p) for p in png_files[1:]]

    # Number of rows
    n_images = len(images)
    n_rows = ceil(n_images / n_cols)

    # Output canvas size
    out_w = tile_w * n_cols
    out_h = tile_h * n_rows

    # Create canvas
    canvas = Image.new("RGB", (out_w, out_h), color=background_color)

    # Paste images
    for idx, im in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols

        # Resize each image to exactly tile size (no gaps)
        tile = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)

        # Paste tile to canvas
        x0 = col * tile_w
        y0 = row * tile_h
        canvas.paste(tile, (x0, y0))

    # Resize final canvas if one side exceeds max_size
    max_size = 10000
    if max(out_w, out_h) > max_size:
        scale = max_size / max(out_w, out_h)
        new_w = int(out_w * scale)
        new_h = int(out_h * scale)
        canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Save result
    out_path = folder_path / output_name
    canvas.save(out_path)
    print(f"Saved tiled image to: {out_path}")


if __name__ == "__main__":
    # Use raw string literal for UNC path
    #folder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260315\KH2PO4_920ex_IRF\IRF"
    folder = r"\\RY-LAB-WS04\ImagingData\Tetsuya\20260315\KH2PO4_920ex_IRF\GFP_2d"
    # Example: use default tile size (150 x 150 pixels)
    make_tiled_png(folder)

    # If you want to enforce a specific tile size, specify resize_to like below
    # make_tiled_png(folder, resize_to=(150, 150))