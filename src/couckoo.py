import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

from ImageProcessor import ImageProcessor
from LSHProcessor import LSHProcessor


# helper functions
def get_image_files(input_dir: str) -> List[str]:
    """
    Retrieve image files from a directory.

    Args:
        input_dir (str): Directory containing images.

    Returns:
        List[str]: List of image file paths.
    """
    image_extensions = (".png", ".jpg", ".jpeg")
    try:
        file_list = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(image_extensions)
        ]
        return file_list
    except FileNotFoundError:
        logging.error(f"Directory not found: {input_dir}")
        raise ValueError(f"Directory {input_dir} does not exist")


def process_images(
    hash_size, bands, image_processor: ImageProcessor, file_list: List[str]
) -> LSHProcessor:
    """
    Process images and calculate their signatures using ImageProcessor.

    Args:
        image_processor (ImageProcessor): Instance of ImageProcessor.
        file_list (List[str]): List of image file paths.

    Returns:
        LSHProcessor: Instance of LSHProcessor populated with image signatures.
    """
    lsh_processor = LSHProcessor(hash_size, bands)
    for file_path in file_list:
        signature = image_processor.calculate_signature(file_path)
        lsh_processor.add_signature(file_path, signature)
    return lsh_processor


def find_duplicates(
    input_dir: str, threshold: float, hash_size: int, bands: int, gen_socres: bool
) -> Tuple[Dict[str, int], List[Tuple[str, str, float]]]:
    """
    Find near-duplicate images within a directory using Locality Sensitive Hashing.

    Args:
        input_dir (str): Directory containing images.
        threshold (float): Similarity threshold.
        hash_size (int): Size of the image hash.
        bands (int): Number of bands for LSH.

    Returns:
        Dict[str, int]: Dictionary of image file paths and their assigned labels.
    """
    image_processor = ImageProcessor(hash_size)
    similarity_scores = []

    try:
        file_list = get_image_files(input_dir)
        if not file_list:
            raise ValueError(f"No valid images found in directory {input_dir}")

        lsh_processor = process_images(hash_size, bands, image_processor, file_list)
        labels = lsh_processor.assign_labels(threshold)

        if gen_socres:
            similarity_scores = lsh_processor.get_similarity_scores(threshold)

        return labels, similarity_scores

    except ValueError as ve:
        logging.error(str(ve))
        return {}, []


def get_results(
    input_dir: str, threshold: float, hash_size: int, bands: int, gen_socres: bool
) -> None:
    """
    outputs a  csv of file names and labels

    Parameters
    ----------
    input_dir (str) : images directory path
    threshold (float) : similarity threshold
    hash_size (int) : hash_size
    bands (int) : band size
    """

    output_file = "results/labels.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    labels, similarity_scores = find_duplicates(
        input_dir, threshold, hash_size, bands, gen_socres
    )
    df = pd.DataFrame(list(labels.items()), columns=["filename", "label"])
    df.sort_values("label").to_csv(output_file)

    if gen_socres:
        generate_similarity_scores(
            similarity_scores,
        )


def generate_similarity_scores(similarity_scores: List[Tuple[str, str, float]]) -> None:
    """
    outputs a  csv of  images file paths and similarity scores


    """
    scores_file = "results/scores.csv"
    os.makedirs(os.path.dirname(scores_file), exist_ok=True)
    df_scores = pd.DataFrame(
        similarity_scores, columns=["imageA", "imageB", "similarity"]
    )
    df_scores.to_csv(scores_file, index=False)


def main(argv):
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Efficient detection of near-duplicate images using locality sensitive hashing"
    )
    args = parser.parse_args()
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing image files.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold for near duplicates.",
    )
    parser.add_argument(
        "-s",
        "--hash_size",
        type=int,
        default=16,
        help="Size of the hash.",
    )
    parser.add_argument("-b", "--bands", type=int, default=16, help="Number of bands.")
    parser.add_argument(
        "-c",
        "--scores",
        type=bool,
        default=True,
        help="generate a duplicates.csv file with duplicated images and the similarity score.",
    )
    parser.add_argument(
        "-l",
        "--gen_lables",
        type=bool,
        default=True,
        help="genrate results.csv file with all near duplicate images having the same label.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    threshold = args.threshold
    hash_size = args.hash_size
    bands = args.bands
    show_similarity_scores = args.scores

    get_results(input_dir, threshold, hash_size, bands, show_similarity_scores)


if __name__ == "__main__":
    main(sys.argv)
