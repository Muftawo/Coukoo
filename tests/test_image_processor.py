import numpy as np
import pytest
from PIL import Image

from ImageProcessor import ImageProcessor


class TestImageProcessor:
    @pytest.fixture
    def image_processor(self):
        return ImageProcessor(hash_size=8)

    @pytest.fixture
    def test_image(self):
        return Image.open("test/data/test_image.jpg")

    def test_flip_to_brightest_quarter(self, image_processor, test_image):
        flipped_image = image_processor.flip_to_brightest_quarter(test_image)
        # Assert that the flipped image is not the same as the original
        assert flipped_image != test_image

    def test_calculate_signature_valid_image(self, image_processor, test_image):
        signature = image_processor.calculate_signature(test_image)
        assert signature is not None
        assert isinstance(signature, np.ndarray)
        assert signature.shape == (64,)  # Expected shape for hash_size=8

    def test_calculate_signature_invalid_image(self, image_processor):
        invalid_image_path = "non/existent/test_image.jpg"
        signature = image_processor.calculate_signature(invalid_image_path)
        assert signature is None

    def test_calculate_signature_unsupported_format(self, image_processor):
        # Replace with an unsupported image format
        unsupported_image_path = "unsupported/format/test_image.txt"
        signature = image_processor.calculate_signature(unsupported_image_path)
        assert signature is None
