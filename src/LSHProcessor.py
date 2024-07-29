import logging
from typing import Dict, List, Tuple

import numpy as np


class LSHProcessor:
    """
    Locality Sensitive Hashing Processor for image similarity detection.
    """

    def __init__(self, hash_size: int, bands: int):
        """
        Initialize LSHProcessor with hash size and number of bands.

        Args:
            hash_size (int): Size of the image hash.
            bands (int): Number of bands for LSH.
        """
        self.hash_size = hash_size
        self.bands = bands
        self.rows = hash_size**2 // bands
        self.hash_buckets_list = [{} for _ in range(bands)]
        self.signatures: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, int] = {}
        self.label_counter = 0
        self.similarity_scores: List[Tuple[str, str, float]] = []

    def add_signature(self, file_path: str, signature: np.ndarray):
        """
        Add image signature to LSH buckets.

        Args:
        file_path (str): File path of the image.
        signature (np.ndarray): Image signature as Numpy n-dimensional array.
        """
        if signature is None:
            return
        self.signatures[file_path] = np.packbits(signature)

        for i in range(self.bands):
            signature_band = signature[i * self.rows : (i + 1) * self.rows]
            signature_band_bytes = signature_band.tobytes()
            if signature_band_bytes not in self.hash_buckets_list[i]:
                self.hash_buckets_list[i][signature_band_bytes] = []
            self.hash_buckets_list[i][signature_band_bytes].append(file_path)

    def calculate_similarity(self, pair: Tuple[str, str]) -> Tuple[str, str, float]:
        """
        Calculate the similarity between two images using hamming distance.

        Args:
            pair (tuble): images to cal similarity for.

        Returns:
            images and the similarity score between them on a (0-1) scale,
            1 being highly similar
        """
        img_a, img_b = pair
        try:
            hd = np.count_nonzero(
                np.unpackbits(self.signatures[img_a])
                != np.unpackbits(self.signatures[img_b])
            )
            similarity = (self.hash_size**2 - hd) / self.hash_size**2
            return img_a, img_b, similarity
        except KeyError:
            logging.error(f"Signatures not found for pair: {pair}")
            return img_a, img_b, 0.0

    def process_similarities(
        self, threshold: float, collect_scores: bool = False
    ) -> None:
        """
        Process and assign labels or collect similarity scores based on threshold.

        Args:
            threshold (float): Similarity threshold to consider images as similar.
            collect_scores (bool): Flag to indicate if similarity scores should be collected.
        """
        for hash_buckets in self.hash_buckets_list:
            for matched_imgs in hash_buckets.values():
                if len(matched_imgs) > 1:
                    for image_a, image_b in zip(matched_imgs, matched_imgs[1:]):
                        img_a, img_b, similarity = self.calculate_similarity(
                            (image_a, image_b)
                        )
                        if similarity >= threshold:
                            if img_a not in self.labels:
                                self.labels[img_a] = self.label_counter
                                self.label_counter += 1
                            if img_b not in self.labels:
                                self.labels[img_b] = self.labels[img_a]

                            if collect_scores:
                                self.similarity_scores.append(
                                    (img_a, img_b, similarity)
                                )

    def assign_labels(self, threshold: float) -> Dict[str, int]:
        """
        Assign integer labels to images, with similar images above threshold having same label.

        Args:
            threshold (float): Similarity threshold to consider images as similar.

        Returns:
            Dict[str, int]: Mapping of image file paths to their assigned labels.
        """
        self.process_similarities(threshold)
        self._assign_labels_remaining_images()
        return self.labels

    def _assign_labels_remaining_images(self) -> None:
        """Assign labels to remaining images (not part of any near-duplicate pair)"""

        for file_path in self.signatures.keys():
            if file_path not in self.labels:
                self.labels[file_path] = self.label_counter
                self.label_counter += 1

    def get_similarity_scores(self, threshold: float) -> List[Tuple[str, str, float]]:
        """
        Updates similar images with their similarity score.

        Args:
            threshold (float): Similarity threshold to consider images as similar.

        """
        self.process_similarities(threshold, collect_scores=True)
        return self.similarity_scores
