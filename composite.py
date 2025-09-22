import argparse
import shutil
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from rasterio import features

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)

DTYPE = np.uint16
COMPRESS = ["COMPRESS=LZW", "PREDICTOR=2"]
NODATA = 0
CRS = 2154


def create_composite(img_paths: list, composite_path: Path):
    """Create composite"""

    # Init a dataframe to hold the composite metadata
    gdf = gpd.GeoDataFrame(columns=["name", "date", "geometry"], crs=CRS)

    # Read first image to initialize composite result
    with rasterio.open(img_paths[0]) as src:
        cols, rows = src.width, src.height
        nbands = src.count

    # Define metadata
    profile = {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": nbands,
        "dtype": "uint16",
        "crs": src.crs,  # Assuming src is a rasterio opened file
        "transform": src.transform,
        "nodata": NODATA,
    }
    profile.update({"compress": "lzw"})
    #  Photometric interpretation
    if composite_path.name.endswith("PXS.tif") or composite_path.name.endswith(
        "RGB.tif"
    ):
        profile["photometric"] = "RGB"

    # Stack and create composite
    imgs = []
    for path in img_paths:
        with rasterio.open(path) as src:
            transform = src.transform
            img = src.read()
            img = np.transpose(img, (1, 2, 0))

        img = np.atleast_3d(img)
        nodata_mask = (img.sum(axis=2) == 0).astype(bool)

        # Mask nodata
        img = img.astype(np.float32)  # Use float16 to allow NaN
        img[nodata_mask] = np.nan

        imgs.append(img)

        # Vectorize nodata_mask
        nodata_mask = ~nodata_mask
        nodata_mask = nodata_mask.astype(np.uint8)
        results = features.shapes(nodata_mask, transform=transform)
        geoms = [shapely.geometry.shape(geom) for geom, val in results if val == 1]
        geoms = [geom.simplify(0.1, preserve_topology=True) for geom in geoms]
        unified = shapely.ops.unary_union(geoms)

        # Add to gdf
        date = path.name.split("_")[2]
        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        tmp_gdf = gpd.GeoDataFrame(
            {"name": path.name, "date": formatted_date, "geometry": [unified]}, crs=CRS
        )
        gdf = pd.concat([gdf, tmp_gdf], ignore_index=True)

    stack = np.stack(imgs, axis=0)

    with warnings.catch_warnings():
        # Ignore mean of empty slice warning as it is expected when all images are nodata
        warnings.simplefilter("ignore", category=RuntimeWarning)
        composite = np.nanmean(stack, axis=0)

    composite = np.atleast_3d(composite)
    composite = np.nan_to_num(composite, nan=0).astype(DTYPE)

    # Write the composite
    with rasterio.open(composite_path, "w", **profile) as dst:
        for bn in range(nbands):
            dst.write(composite[:, :, bn], bn + 1)
            dst.update_tags(bn + 1, nodata=NODATA)

    gdf.to_file(composite_path.parent / f"{composite_path.stem}_metadata.gpkg")


def main(config_file: str):
    """Main process"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

        # Create composite main dir if not exists
        output_dir = Path(config["composites_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)

        # Copy config in output directory
        shutil.copy(config_file, config["composites_dir"])

        # Get images
        img_dir = Path(config["img_dir"])
        img_paths = list(img_dir.glob("*/*.tif"))

        for composite in config["composites"]:
            print(f"Create composite {composite}")

            composite_dir = output_dir / composite
            composite_dir.mkdir(exist_ok=True, parents=True)
            products = config["composites"][composite]

            # Get composite image paths
            product_paths = []
            for product in products:
                product_paths.extend(path for path in img_paths if product in path.name)

            # Create composite per image type
            for img_type in ["PAN", "RGB", "PXS"]:
                composite_img_paths = [
                    path for path in product_paths if img_type in path.name
                ]

                composite_path = composite_dir / f"{composite}_{img_type}.tif"
                create_composite(composite_img_paths, composite_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create image composites from a config file."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the configuration file containing composite definitions.",
    )
    args = parser.parse_args()

    main(args.config_file)
