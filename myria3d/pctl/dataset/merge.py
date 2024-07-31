from typing import List
import argparse
import pdal
import os
import numpy as np
from pdaltools import las_info
from glob import glob
from pathlib import Path

from .utils import get_pdal_reader, pdal_read_las_array_parallel


def merge_features_and_classification(features_path: str, base_path: str, filename: str, scales: List[str], cloud_outpath: str, epsg: str = "4326"):
    base_filename = os.path.join(base_path, f"{filename}.laz")
    print(f"Reading PC {base_filename} with classification")
    arr_class = pdal_read_las_array_parallel(base_filename, epsg, extra_dims='all', num_blocks=16)

    arr_feats = []
    feats_dt = []
    for scale in scales:
        filepath = os.path.join(features_path, filename, f"{scale}.laz")
        print(f"Reading PC {filepath} with features")
        arr_feats.append(pdal_read_las_array_parallel(filepath, epsg, extra_dims='all', num_blocks=16))
        feats_dt += [(f'Linearity_{scale}', '<f8'),
                     (f'Planarity_{scale}', '<f8'),
                     (f'Scattering_{scale}', '<f8'),
                     (f'EigenvalueSum_{scale}', '<f8'),
                     (f'Eigenentropy_{scale}', '<f8'),
                     (f'Omnivariance_{scale}', '<f8')]

    new_dt = np.dtype(arr_class.dtype.descr + feats_dt)

    print("Merging Fields...")
    merged_arr = np.zeros(arr_class.shape, dtype=new_dt)

    # copying base input cloud
    for descr in arr_class.dtype.descr:
        merged_arr[descr[0]] = arr_class[descr[0]]

    # copying scales clouds
    for scale, arr_feat in zip(scales, arr_feats):
        for descr in ['Linearity', 'Planarity', 'Scattering', 'EigenvalueSum', 'Eigenentropy', 'Omnivariance']:
            merged_arr[f'{descr}_{scale}'] = arr_feat[descr]

    pipeline = pdal.Pipeline() | get_pdal_reader(base_filename, epsg)
    pipeline.execute()
    print("Saving file ...")
    writer_params = las_info.get_writer_parameters_from_reader_metadata(
        pipeline.metadata, a_srs=f"EPSG:{epsg}" if str(epsg).isdigit() else epsg
    )
    writer_params["extra_dims"] = "all"

    out_pipe = pdal.Writer.las(filename=cloud_outpath, **writer_params).pipeline(merged_arr)
    out_pipe.execute()


def main(classif_basedir, feats_basedir, output_basedir, scales: List[str], extension: str = '.laz'):
    if not os.path.exists(output_basedir):
        os.makedirs(output_basedir, exist_ok=True)

    files = glob(os.path.join(classif_basedir, f'*{extension}'))
    files.sort()
    for i, f in enumerate(files):
        cloud_outpath = os.path.join(output_basedir, Path(f).name)
        print(f"{i} -> Treating {f}")
        merge_features_and_classification(base_path=classif_basedir,
                                          features_path=feats_basedir,
                                          filename=Path(f).stem,
                                          scales=scales,
                                          cloud_outpath=cloud_outpath)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Application to merge point clouds with"
                                     " features with the ones containing the classification")

    parser.add_argument("--class-dir", "-c", help="Directory containing pcs with classification info")
    parser.add_argument("--features-dir", "-f", help="Directory containing pcs with features info")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--scales", "-s", type=str, nargs='+', help='List of scales')
    parser.add_argument("--extension", "-e", type=str, default=".laz", choices=[".laz", ".las", ".e57"], help='Extension of point clouds')

    args = parser.parse_args()
    print(args)
    classif_dir = args.class_dir
    feats_dir = args.features_dir
    output_dir = args.output_dir
    scales = args.scales
    extension = args.extension

    main(classif_dir, feats_dir, output_dir, scales, extension)

