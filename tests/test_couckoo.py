import os

import numpy as np
import pytest

from src.couckoo import find_duplicates, get_image_files, process_images
from src.ImageProcessor import ImageProcessor
from src.LSHProcessor import LSHProcessor


class TestCouckoo:
    @pytest.fixture
    def image_processor(self):
        return ImageProcessor(hash_size=8)

    @pytest.fixture
    def lsh_processor(self):
        return LSHProcessor(hash_size=8, bands=4)

    def test_get_image_files_valid_directory(self):
        """Test that get_image_files returns a list of image files for a valid directory."""
        input_dir = (
            "tests/data"  # Replace with the actual path to your test data directory
        )
        file_list = get_image_files(input_dir)
        assert isinstance(file_list, list)
        assert len(file_list) > 0
        for file_path in file_list:
            assert os.path.isfile(file_path)
            assert file_path.lower().endswith((".png", ".jpg", ".jpeg"))

    def test_get_image_files_empty_directory(self):
        """Test that get_image_files returns an empty list for an empty directory."""
        input_dir = (
            "tests/empty_dir"  # Replace with the actual path to an empty directory
        )
        file_list = get_image_files(input_dir)
        assert file_list == []

    def test_get_image_files_invalid_directory(self):
        """Test that get_image_files raises a ValueError for an invalid directory."""
        input_dir = "tests/nonexistent_dir"  # Replace with an invalid directory path
        with pytest.raises(ValueError):
            get_image_files(input_dir)

    def test_get_image_files_directory_with_non_image_files(self):
        """Test that get_image_files only returns image files from a directory with non-image files."""
        input_dir = "tests/data_with_non_image_files"  # Replace with the actual path to your test data directory
        file_list = get_image_files(input_dir)
        assert isinstance(file_list, list)
        assert len(file_list) > 0
        for file_path in file_list:
            assert os.path.isfile(file_path)
            assert file_path.lower().endswith((".png", ".jpg", ".jpeg"))

    def test_process_images(self, image_processor, lsh_processor, monkeypatch):
        """Test that process_images populates the LSHProcessor with image signatures."""
        file_list = [
            "tests/data/image1.jpg",
            "tests/data/image2.jpg",
        ]  # Replace with actual paths
        monkeypatch.setattr(
            image_processor,
            "calculate_signature",
            lambda x: np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]),
        )
        processed_lsh_processor = process_images(8, 4, image_processor, file_list)
        assert len(processed_lsh_processor.signatures) == 2
        assert "tests/data/image1.jpg" in processed_lsh_processor.signatures
        assert "tests/data/image2.jpg" in processed_lsh_processor.signatures

    def test_find_duplicates_valid_directory(
        self, image_processor, lsh_processor, monkeypatch
    ):
        """Test that find_duplicates returns labels and similarity scores for a valid directory."""
        input_dir = (
            "tests/data"  # Replace with the actual path to your test data directory
        )
        monkeypatch.setattr(
            lsh_processor,
            "assign_labels",
            lambda x: {"tests/data/image1.jpg": 0, "tests/data/image2.jpg": 0},
        )
        monkeypatch.setattr(
            lsh_processor,
            "get_similarity_scores",
            lambda x: [("tests/data/image1.jpg", "tests/data/image2.jpg", 0.8)],
        )
        labels, similarity_scores = find_duplicates(input_dir, 0.8, 8, 4, True)
        assert labels == {"tests/data/image1.jpg": 0, "tests/data/image2.jpg": 0}
        assert similarity_scores == [
            ("tests/data/image1.jpg", "tests/data/image2.jpg", 0.8)
        ]

    def test_find_duplicates_empty_directory(self):
        """Test that find_duplicates returns empty labels and similarity scores for an empty directory."""
        input_dir = (
            "tests/empty_dir"  # Replace with the actual path to an empty directory
        )
        labels, similarity_scores = find_duplicates(input_dir, 0.8, 8, 4, True)
        assert labels == {}
        assert similarity_scores == []

    def test_find_duplicates_invalid_directory(self):
        """Test that find_duplicates raises a ValueError for an invalid directory."""
        input_dir = "tests/nonexistent_dir"  # Replace with an invalid directory path
        with pytest.raises(ValueError):
            find_duplicates(input_dir, 0.8, 8, 4, True)

    def test_find_duplicates_no_valid_images(self, monkeypatch):
        """Test that find_duplicates returns empty labels and similarity scores if no valid images are found."""
        input_dir = (
            "tests/data"  # Replace with the actual path to your test data directory
        )
        monkeypatch.setattr(get_image_files, "return_value", [])
        labels, similarity_scores = find_duplicates(input_dir, 0.8, 8, 4, True)
        assert labels == {}
        assert similarity_scores == []
