import glob
import json
from pathlib import Path
import subprocess as sp
import multiprocessing as mp
from numbers import Number
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import pdal
from scipy.spatial import cKDTree

SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
LAS_PATHS_BY_SPLIT_DICT_TYPE = Dict[SPLIT_TYPE, List[str]]


def find_file_in_dir(data_dir: str, basename: str) -> str:
    """Query files matching a basename in input_data_dir and its subdirectories.
    Args:
        input_data_dir (str): data directory
    Returns:
        [str]: first file path matching the query.
    """
    query = f"{data_dir}/**/{basename}"
    files = glob.glob(query, recursive=True)
    return files[0]


def get_mosaic_of_centers(tile_width: Number, subtile_width: Number, subtile_overlap: Number = 0):
    if subtile_overlap < 0:
        raise ValueError("datamodule.subtile_overlap must be positive.")

    xy_range = np.arange(
        subtile_width / 2,
        tile_width + (subtile_width / 2) - subtile_overlap,
        step=subtile_width - subtile_overlap,
    )
    return [np.array([x, y]) for x in xy_range for y in xy_range]


def pdal_read_las_array(las_path: str, epsg: str, extra_dims: str = None, index: int = None, count: int = None):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path
        epsg (str): epsg to force the reading with

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.

    """
    p1 = pdal.Pipeline() | get_pdal_reader(las_path, epsg, extra_dims, index, count)
    p1.execute()
    return p1.arrays[0]


def pdal_read_las_array_as_float32(las_path: str, epsg: str, extra_dims: str, index: int = None, count: int = None, offset: np.array = None) -> np.array:
    """Read LAS as a a named array, casted to floats."""
    arr = pdal_read_las_array(las_path, epsg, extra_dims, index, count)

    if offset is None:
        offset = np.array([arr['X'][0], arr['Y'][0], arr['Z'][0]])

    arr['X'] -= offset[0]
    arr['Y'] -= offset[1]
    arr['Z'] -= offset[2]

    all_floats = np.dtype({"names": arr.dtype.names, "formats": ["f4"] * len(arr.dtype.names)})
    out = arr.astype(all_floats)

    return out

def get_metadata(las_path: str) -> dict:
    """ returns metadata contained in a las file
    Args:
        las_path (str): input LAS path to get metadata from.
    Returns:
        dict : the metadata.
    """
    pipeline = pdal.Reader.las(filename=las_path).pipeline()
    pipeline.execute()
    return pipeline.metadata


def get_pdal_reader(las_path: str, epsg: str, extra_dims: str = None, start=None, count=None) -> pdal.Reader.las:
    """Standard Reader.
    Args:
        las_path (str): input LAS path to read.
        epsg (str): epsg to force the reading with
    Returns:
        pdal.Reader.las: reader to use in a pipeline.

    """
    if epsg :
        reader_args = dict(
            filename=las_path,
            nosrs=True,
            override_srs=f"EPSG:{epsg}" if str(epsg).isdigit() else epsg,
            use_eb_vlr=True)

        if start is not None:
            reader_args["start"] = start
        if count is not None:
            reader_args["count"] = count

        # if an epsg in provided, force pdal to read the lidar file with it
        # epsg can be added as a number like "2154" or as a string like "EPSG:2154"
        reader = pdal.Reader.las(**reader_args)

        if extra_dims:
            reader.extra_dims = extra_dims

        return reader

    try :
        if get_metadata(las_path)['metadata']['readers.las']['srs']['compoundwkt']:
            # read the lidar file with pdal default
            return pdal.Reader.las(filename=las_path)
    except Exception:
        pass  # we will go to the "raise exception" anyway

    raise Exception("No EPSG provided, neither in the lidar file or as parameter")


def get_pdal_info_metadata(las_path: str) -> Dict:
    """Read las metadata using pdal info
    Args:
        las_path (str): input LAS path to read.
    Returns:
        (dict): dictionary containing metadata from the las file
    """
    r = sp.run(["pdal", "info", "--metadata", las_path, '--readers.las.nosrs=true'], capture_output=True)
    if r.returncode == 1:
        msg = r.stderr.decode()
        raise RuntimeError(msg)

    output = r.stdout.decode()
    json_info = json.loads(output)

    return json_info["metadata"]


def process_block_float32(args):
    """Wrapper function for multiprocessing."""
    las_path, epsg, extra_dims, index, count, offset = args
    return pdal_read_las_array_as_float32(las_path, epsg, extra_dims, index, count, offset=offset)

def process_block(args):
    """Wrapper function for multiprocessing."""
    las_path, epsg, extra_dims, index, count = args
    return pdal_read_las_array(las_path, epsg, extra_dims, index, count)

# hdf5, iterable
def pdal_read_las_array_as_float32_parallel(las_path, epsg, extra_dims=None, num_blocks=4):
    metadata = get_pdal_info_metadata(las_path)
    num_points = metadata["count"]
    offset_x = metadata["offset_x"]
    offset_y = metadata["offset_y"]
    offset_z = metadata["offset_z"]
    offset = np.array([offset_x, offset_y, offset_z], dtype=np.float32)

    # Calculate the number of points per block
    points_per_block = num_points // num_blocks

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Prepare arguments for each process
        args = [(las_path, epsg, extra_dims, i * points_per_block, points_per_block, offset) for i in range(num_blocks)]

        # Process the last block separately to include any remaining points
        if num_points % num_blocks != 0:
            args.append((las_path, epsg, extra_dims, num_blocks * points_per_block, num_points % points_per_block, offset))

        # Map the process_block function to the list of arguments
        results = pool.map(process_block_float32, args)
    # Combine the results
    combined = np.concatenate(results)

    return combined


def pdal_read_las_array_parallel(las_path, epsg, extra_dims=None, num_blocks=4):
    metadata = get_pdal_info_metadata(las_path)
    num_points = metadata["count"]
    # Calculate the number of points per block
    points_per_block = num_points // num_blocks

    # Create a pool of worker processes
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Prepare arguments for each process
        args = [(las_path, epsg, extra_dims, i * points_per_block, points_per_block) for i in range(num_blocks)]

        # Process the last block separately to include any remaining points
        if num_points % num_blocks != 0:
            args.append((las_path, epsg, extra_dims, num_blocks * points_per_block, num_points % points_per_block))

        # Map the process_block function to the list of arguments
        results = pool.map(process_block, args)
    # Combine the results
    combined = np.concatenate(results)

    return combined


def split_cloud_into_samples(
    las_path: str,
    tile_width: Number,
    subtile_width: Number,
    epsg: str,
    subtile_overlap: Number = 0,
    min_num_of_points: Number = 512,
    extra_dims=None,
    num_blocks: int = 16
):
    """Split LAS point cloud into samples.

    Args:
        las_path (str): path to raw LAS file
        tile_width (Number): width of input LAS file
        subtile_width (Number): width of receptive field.
        epsg (str): epsg to force the reading with
        subtile_overlap (Number, optional): overlap between adjacent tiles. Defaults to 0.

    Yields:
        _type_: idx_in_original_cloud, and points of sample in pdal input format casted as floats.

    """
    if num_blocks == 1:
        points = pdal_read_las_array_as_float32(las_path, epsg, extra_dims)
    else:
        points = pdal_read_las_array_as_float32_parallel(las_path, epsg, extra_dims, num_blocks=num_blocks)
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    kd_tree = cKDTree(pos[:, :2] - pos[:, :2].min(axis=0))
    XYs = get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap=subtile_overlap)
    for center in XYs:
        radius = subtile_width // 2  # Square receptive field.
        minkowski_p = np.inf
        sample_idx = np.array(kd_tree.query_ball_point(center, r=radius, p=minkowski_p))
        if len(sample_idx) < min_num_of_points:
            # no points in this receptive fields
            continue
        sample_points = points[sample_idx]
        yield sample_idx, sample_points


def pre_filter_below_n_points(data, min_num_nodes=1):
    return data.pos.shape[0] < min_num_nodes


def get_las_paths_by_split_dict(
    data_dir: str, split_csv_path: str
) -> LAS_PATHS_BY_SPLIT_DICT_TYPE:
    las_paths_by_split_dict: LAS_PATHS_BY_SPLIT_DICT_TYPE = {}
    split_df = pd.read_csv(split_csv_path)
    for phase in ["train", "val", "test"]:
        basenames = split_df[split_df.split == phase].basename.tolist()
        # Reminder: an explicit data structure with ./val, ./train, ./test subfolder is required.
        las_paths_by_split_dict[phase] = [str(Path(data_dir) / phase / b) for b in basenames]

    if not las_paths_by_split_dict:
        raise FileNotFoundError(
            (
                f"No basename found while parsing directory {data_dir}"
                f"using {split_csv_path} as split CSV."
            )
        )

    return las_paths_by_split_dict
