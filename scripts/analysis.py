"""beetle_feeding_damage analysis."""

import os
import logging
import argparse
import errno

import numpy as np

from dtoolcore import DataSet

from jicbioimage.core.image import Image
from jicbioimage.core.transform import transformation
from jicbioimage.core.io import AutoName, AutoWrite

from skimage.morphology import disk

from jicbioimage.transform import (
    invert,
    threshold_otsu,
    remove_small_objects,
    erode_binary,
)

from jicbioimage.segment import (
    watershed_with_seeds,
    connected_components,
    Region,
)

from jicbioimage.illustrate import Canvas

__version__ = "0.1.0"

AutoName.prefix_format = "{:03d}_"


@transformation
def identity(image):
    """Return the image as is."""
    return image


@transformation
def select_red(image):
    return image[:, :, 0]


@transformation
def threshold_abs(image, min_value):
    return image > min_value


@transformation
def fill_small_holes(image, min_size):
    aw = AutoWrite.on
    AutoWrite.on = False
    image = invert(image)
    image = remove_small_objects(image, min_size=min_size)
    image = invert(image)
    AutoWrite.on = aw
    return image


def fill_small_holes_in_region(region, min_size):
    aw = AutoWrite.on
    AutoWrite.on = False
    region = invert(region)
    region = remove_small_objects(region, min_size=min_size)
    region = invert(region)
    AutoWrite.on = aw
    return region


def analyse_file_org(fpath, output_directory):
    """Analyse a single file."""
    logging.info("Analysing file: {}".format(fpath))
    image = Image.from_file(fpath)
    image = identity(image)
    image = select_red(image)
    image = invert(image)
    image = threshold_otsu(image)

    seeds = remove_small_objects(image, min_size=1000)
    seeds = fill_small_holes(seeds, min_size=1000)
    seeds = erode_binary(seeds, selem=disk(30))
    seeds = connected_components(seeds, background=0)

    watershed_with_seeds(-image, seeds=seeds, mask=image)


def get_negative_single_channel(image):
    negative = identity(image)
    negative = select_red(negative)
    negative = invert(negative)
    return negative


def find_seeds(image):
    seeds = threshold_abs(image, 200)
    seeds = remove_small_objects(seeds, min_size=1000)
    seeds = connected_components(seeds, background=0)
    return seeds


def find_mask(image):
    mask = threshold_abs(image, 170)
    return mask


@transformation
def post_process_segmentation(segmentation):
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        region = fill_small_holes_in_region(region, 10000)
        segmentation[region] = i
    return segmentation


def process_leaf(whole_leaf, eaten_leaf, ann):
    ys, xs = whole_leaf.index_arrays
    bounding_box = np.zeros(ann.shape[0:2]).view(Region)
    bounding_box[np.min(ys):np.max(ys), np.min(xs):np.max(xs)] = True

    eaten_fraction = float(eaten_leaf.area) / float(whole_leaf.area)
    percentage_eaten = (1.0 - eaten_fraction) * 100

    ann.mask_region(eaten_leaf.inner.border.dilate(), (255, 0, 255))
    ann.mask_region(whole_leaf.inner.border.dilate(), (0, 255, 255))
    ann.mask_region(bounding_box.inner.border.dilate(), (255, 0, 0))

    ann.text_at(
        "{:.0f}%".format(percentage_eaten),
        (np.min(ys) + 10, np.min(xs) + 10),
        size=56)

    return ann


@transformation
def annotate(image, whole_leaf_segmentation, eaten_leaf_segmentation):
    ann = image.view(Canvas)
    for i in whole_leaf_segmentation.identifiers:
        whole_leaf_region = whole_leaf_segmentation.region_by_identifier(i)
        eaten_leaf_region = eaten_leaf_segmentation.region_by_identifier(i)
        ann = process_leaf(whole_leaf_region, eaten_leaf_region, ann)
    return ann


def analyse_file(fpath, output_directory, test_data_only=False):
    """Analyse a single file."""
    logging.info("Analysing file: {}".format(fpath))
    AutoName.directory = output_directory

    image = Image.from_file(fpath)

    negative = get_negative_single_channel(image)
    seeds = find_seeds(negative)
    mask = find_mask(negative)

    eaten_leaf_segmentation = watershed_with_seeds(
        negative, seeds=seeds, mask=mask)
    whole_leaf_segmentation = post_process_segmentation(
        eaten_leaf_segmentation.copy())

    ann = annotate(image, whole_leaf_segmentation, eaten_leaf_segmentation)
    ann_fpath = os.path.join(output_directory, "annotated.png")
    with open(ann_fpath, "wb") as fh:
        fh.write(ann.png())


def safe_mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(directory):
            pass
        else:
            raise


def data_item_directory(output_directory, rel_path):
    abs_path = os.path.join(output_directory, rel_path)
    safe_mkdir(abs_path)
    return abs_path


def analyse_dataset(dataset_dir, output_dir, test_data_only=False):
    """Analyse all the files in the dataset."""
    dataset = DataSet.from_path(dataset_dir)
    logging.info("Analysing files in dataset: {}".format(dataset.name))

    for i in dataset.identifiers:
        abs_path = dataset.abspath_from_identifier(i)
        item_info = dataset.item_from_identifier(i)

        specific_output_dir = data_item_directory(
            output_dir, item_info["path"])
        analyse_file(abs_path, specific_output_dir, test_data_only)
        if test_data_only:
            break


def analyse_directory(input_directory, output_directory):
    """Analyse all the files in a directory."""
    logging.info("Analysing files in directory: {}".format(input_directory))
    for fname in os.listdir(input_directory):
        fpath = os.path.join(input_directory, fname)
        analyse_file(fpath, output_directory)


def main():
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_source", help="Input file/directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--debug", default=False, action="store_true",
                        help="Write out intermediate images")
    parser.add_argument("--test", default=False, action="store_true",
                        help="Use only test data")
    args = parser.parse_args()

    # Create the output directory if it does not exist.
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    # Only write out intermediate images in debug mode.
    if not args.debug:
        AutoWrite.on = False

    # Setup a logger for the script.
    log_fname = "audit.log"
    log_fpath = os.path.join(args.output_dir, log_fname)
    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    logging.basicConfig(filename=log_fpath, level=logging_level)

    # Log some basic information about the script that is running.
    logging.info("Script name: {}".format(__file__))
    logging.info("Script version: {}".format(__version__))

    # Run the analysis.
    analyse_dataset(args.input_source, args.output_dir, args.test)


if __name__ == "__main__":
    main()
