# **Coukoo**
Coukoo is an image duplicate detection tool, designed to remove duplicate images from large datasets using locality-sensitive hashing. To achieve image deduplication at scale, an approximating search algorithm such as LSH offers a significant tradeoff for speed and efficiency at the cost of some accuracy, which can be adjusted through parameter tuning.


# Overview
This program employs Locality Sensitive Hashing (LSH) to detect near-duplicate images within a directory. The ability to detect duplicates for deduplication at scale is crucial to maintaining good-quality image datasets. The program is structured into the following components:



1. ### ImageProcessor Class:
   * Handles image preprocessing tasks such as converting images to grayscale, resizing, and flipping to normalize the brightest quarters.
   * Computes a perceptual hash (dhash) of the image to generate a signature for similarity comparison.



2. ### LSHProcessor Class:
   *  Implements LSH for efficient similarity detection by dividing image signatures into bytes.
   *  Stores image signatures in buckets and provides methods to find  potentially similar images.



3. ### Helper Functions:
   *   get_image_files: Retrieves a list of image files from a specified directory based on recognized file extensions.
   *   process_images: Processes each image in the directory using the ImageProcessor and populates the LSHProcessor with image signatures.
   *   find_near_duplicates: Coordinates the entire process by initializing components, and finding image duplicates using LSH.



4. ### Program Execution:

   * Reads input directory path, similarity threshold, hash size, and number of bands.
   * Uses find_near_duplicates to identify near-duplicate images based on the provided threshold.
   * Outputs a CSV file (results.csv) containing filenames and their corresponding similarity labels.


## **Components**
* ImageProcessor: Preprocesses images and computes their signatures.
* LSHProcessor: Implements LSH for efficient similarity detection.
* Utility Functions: Handle file operations and coordinate image processing tasks.


## **Usage**

* Inputs:
  * `input_dir`: Directory path containing images to be analyzed.
  * `threshold`: Minimum similarity threshold (between 0 and 1) for considering images as near-duplicates.
  * `hash_size and bands`: Parameters for LSH configuration, affecting granularity and efficiency of similarity detection.


* Output:

  * Generates a CSV file (results.csv) containing image paths and labels with duplicates having the same label.

## **Duplicate detector pipeline**
For detecting similarity between two images `A` and `B` at a threshold `X`.

  1. The `ImageProcessor` class is used to calculate the image signature/hash with the `calculate_signature` method.

     *  The image is converted to grayscale and resized to `(hash_size+1, hash_size)` scale.
     *  The image is then flipped to ensure the brightest quatre is always at the top left. to deal with image rotations.
     *  A difference hash is then calculated using hash_size, and then collapsed  to a 1-dimensional array.  
     *  This 1-dimensional array is returned as the signature of the image.
  
  2. The `LSHProcessor` class is employed to ;

      * Add each image path and signature to the bucket list, `hash_buckets_list` using the `add_signature` method. The `band size` and `rows` are used to iteratively calculate different signature bytes and stored in the  `hash_buckets_list` if a previous image has produced the same bytes, the image path is append to its list of image paths, in the `hash_buckets_list`. This indicates the current row in the image is similar to the previous row of a different image.
        * NB: `hash_bucket_list` contains dicts of signature bytes as keys and a  list of image paths as values

     *  Assign labels, For each similar image paths list in `hash_bucket_list`, we iteratively compare them to each other in pairs, and calculate a similarity score using the `calculate_similarity` method which uses `hamming distance` to calculate the similarity between image signatures. If the similarity score exceeds the threshold, the same label is assigned to both images. For images that are not assigned any labels through the previous step new labels are assigned.
  
  3. For images  `A` and `B` if their  similarity score exceeds threshold `X`, the same label is assigned.







