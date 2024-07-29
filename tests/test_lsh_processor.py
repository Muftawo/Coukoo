import numpy as np
import pytest

from src.LSHProcessor import LSHProcessor


class TestLSHProcessor:
    @pytest.fixture
    def lsh_processor(self):
        return LSHProcessor(hash_size=8, bands=4)

    def test_add_signature(self, lsh_processor):
        signature = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        file_path = "test/data/test_image.jpg"
        lsh_processor.add_signature(file_path, signature)

        assert file_path in lsh_processor.signatures
        assert np.array_equal(
            lsh_processor.signatures[file_path], np.packbits(signature)
        )

        # Check if the signature is added to the correct hash buckets
        for i in range(lsh_processor.bands):
            signature_band = signature[
                i * lsh_processor.rows : (i + 1) * lsh_processor.rows
            ]
            signature_band_bytes = signature_band.tobytes()
            assert file_path in lsh_processor.hash_buckets_list[i][signature_band_bytes]

    def test_add_signature_none(self, lsh_processor):
        file_path = "test/data/test_image.jpg"
        lsh_processor.add_signature(file_path, None)

        assert file_path not in lsh_processor.signatures

    def test_calculate_similarity(self, lsh_processor):
        lsh_processor.signatures["image1.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8
        )
        lsh_processor.signatures["image2.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8
        )

        img_a, img_b, similarity = lsh_processor.calculate_similarity(
            ("image1.jpg", "image2.jpg")
        )
        assert img_a == "image1.jpg"
        assert img_b == "image2.jpg"
        assert similarity == 15 / 16

    def test_calculate_similarity_not_found(self, lsh_processor):
        img_a, img_b, similarity = lsh_processor.calculate_similarity(
            ("image1.jpg", "image2.jpg")
        )
        assert img_a == "image1.jpg"
        assert img_b == "image2.jpg"
        assert similarity == 0.0

    def test_process_similarities_assign_labels(self, lsh_processor):
        lsh_processor.signatures["image1.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8
        )
        lsh_processor.signatures["image2.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image3.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image4.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8
        )

        lsh_processor.process_similarities(threshold=0.8)
        labels = lsh_processor.labels

        assert labels["image1.jpg"] == labels["image2.jpg"]
        assert labels["image3.jpg"] == labels["image4.jpg"]
        assert labels["image1.jpg"] != labels["image3.jpg"]

    def test_process_similarities_collect_scores(self, lsh_processor):
        lsh_processor.signatures["image1.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8
        )
        lsh_processor.signatures["image2.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image3.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image4.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8
        )

        lsh_processor.process_similarities(threshold=0.8, collect_scores=True)
        similarity_scores = lsh_processor.similarity_scores

        assert len(similarity_scores) == 2
        assert ("image1.jpg", "image2.jpg", 15 / 16) in similarity_scores
        assert ("image3.jpg", "image4.jpg", 15 / 16) in similarity_scores

    def test_assign_labels(self, lsh_processor):
        lsh_processor.signatures["image1.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8
        )
        lsh_processor.signatures["image2.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image3.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image4.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8
        )

        labels = lsh_processor.assign_labels(threshold=0.8)

        assert labels["image1.jpg"] == labels["image2.jpg"]
        assert labels["image3.jpg"] == labels["image4.jpg"]
        assert labels["image1.jpg"] != labels["image3.jpg"]

    def test_get_similarity_scores(self, lsh_processor):
        lsh_processor.signatures["image1.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8
        )
        lsh_processor.signatures["image2.jpg"] = np.array(
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image3.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1], dtype=np.uint8
        )
        lsh_processor.signatures["image4.jpg"] = np.array(
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0], dtype=np.uint8
        )

        similarity_scores = lsh_processor.get_similarity_scores(threshold=0.8)

        assert len(similarity_scores) == 2
        assert ("image1.jpg", "image2.jpg", 15 / 16) in similarity_scores
        assert ("image3.jpg", "image4.jpg", 15 / 16) in similarity_scores
