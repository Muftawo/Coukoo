import logging

import imagehash
import numpy as np
from PIL import Image, ImageOps


class ImageProcessor:
    def __init__(self, hash_size: int):
        self.hash_size = hash_size

    def flip_to_brightest_quarter(self, image):
        """
        Determine which quarter of the image is the brightest and flip accordingly, to ensure normalization and consistency.
        Args:
            image: PIL Image object
        Returns:
            Flipped PIL Image object
        """
        h, w = image.size
        quarters = {
            "top_left": image.crop((0, 0, w // 2, h // 2)),
            "top_right": image.crop((w // 2, 0, w, h // 2)),
            "bottom_left": image.crop((0, h // 2, w // 2, h)),
            "bottom_right": image.crop((w // 2, h // 2, w, h)),
        }
        brightest_quarter = max(quarters, key=lambda x: np.sum(np.array(quarters[x])))

        if brightest_quarter == "top_left":
            return image
        elif brightest_quarter == "top_right":
            return ImageOps.mirror(image)  # horizontal flip
        elif brightest_quarter == "bottom_left":
            return ImageOps.flip(image)  # vertical flip
        elif brightest_quarter == "bottom_right":
            return ImageOps.mirror(
                ImageOps.flip(image)
            )  # vertical then horizontal filp

    def calculate_signature(self, image_file: str) -> np.ndarray:
        """
        Calculate the signature of a given file.

        Args:
            image_file: the image path to calculate the signature for

        Returns:
            Image signature as Numpy n-dimensional array or None if the file is not a PIL recognized image
        """
        try:
            with Image.open(image_file) as pil_image:
                pil_image = pil_image.convert("L").resize(
                    (self.hash_size + 1, self.hash_size), Image.LANCZOS
                )
                flipped_image = self.flip_to_brightest_quarter(pil_image)
                dhash = imagehash.dhash(flipped_image, self.hash_size)
                signature = dhash.hash.flatten()
            return signature
        except FileNotFoundError:
            logging.error(f"File not found: {image_file}")
        except Exception as e:
            logging.error(f"Error processing {image_file}: {e}")
        return None
