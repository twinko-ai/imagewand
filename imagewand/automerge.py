import logging
import os
import glob
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm


def automerge(
    input_paths: Union[str, List[str]],
    output_path: Optional[str] = None,
    debug: bool = False,
) -> str:
    """
    Automatically merge overlapping images into a panorama.

    Args:
        input_paths: List of image paths, directory, or glob pattern
        output_path: Path to save merged result (optional)
        debug: Enable debug mode to show intermediate results

    Returns:
        Path to the merged output image
    """
    # Handle directory input
    if isinstance(input_paths, str):
        if os.path.isdir(input_paths):
            # Directory input
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                image_files.extend(glob.glob(os.path.join(input_paths, ext)))
            input_paths = sorted(image_files)
        elif "*" in input_paths:
            # Glob pattern input
            input_paths = sorted(glob.glob(input_paths))
        else:
            # Single file input
            input_paths = [input_paths]

    if len(input_paths) < 2:
        raise ValueError("At least two images are required for merging")

    # Create output path if not specified
    if output_path is None:
        # Use the directory of the first image
        base_dir = os.path.dirname(input_paths[0])
        if not base_dir:
            base_dir = "."
        output_path = os.path.join(base_dir, "merged_result.jpg")

    # Load images
    images = []
    for path in tqdm(input_paths, desc="Loading images"):
        img = cv2.imread(path)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        raise ValueError("Could not load at least two valid images")

    # Create AutoMerge instance and merge images
    merger = AutoMerge(debug=debug)
    result = merger.merge_images(images)

    # Save the result
    cv2.imwrite(output_path, result)
    return output_path


class AutoMerge:
    def __init__(self, debug: bool = False):
        """
        Initialize the AutoMerge class

        Args:
            debug (bool): Enable debug mode to show intermediate results
        """
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        # SIFT is better for feature detection than ORB
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)

        # Minimum number of good matches required
        self.MIN_MATCHES = 10

    def _detect_and_compute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and compute keypoints and descriptors

        Args:
            image: Input image

        Returns:
            Tuple of keypoints and descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def _find_matches(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Find matches between two sets of descriptors
        """
        # Use k-nearest neighbors matching
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []

        # Apply Lowe's ratio test
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        return good_matches

    def _try_all_orientations(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Try different orientations of the second image to find the best match
        """
        orientations = [
            (img2, "normal"),
            (cv2.flip(img2, 1), "horizontal flip"),
            (cv2.flip(img2, 0), "vertical flip"),
            (cv2.rotate(img2, cv2.ROTATE_180), "180 rotation"),
        ]

        best_matches = []
        best_img = img2
        best_orientation = "normal"

        kp1, desc1 = self._detect_and_compute(img1)

        for oriented_img, orientation in orientations:
            kp2, desc2 = self._detect_and_compute(oriented_img)
            matches = self._find_matches(desc1, desc2)

            if len(matches) > len(best_matches):
                best_matches = matches
                best_img = oriented_img
                best_orientation = orientation

        if self.debug:
            self.logger.info(f"Best orientation: {best_orientation}")

        return kp1, best_img, best_matches

    def _remove_white_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove white borders from an image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # More aggressive threshold for white detection
        _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to remove noise
        kernel = np.ones((7, 7), np.uint8)  # Larger kernel
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # Find contours of non-white regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find the bounding rectangle of all content
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Add small padding
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        x_max = min(image.shape[1], x + w + padding)
        y_max = min(image.shape[0], y + h + padding)

        # Additional check for white strips
        for edge in ["top", "bottom", "left", "right"]:
            if edge == "top":
                region = gray[y : y + 10, x:x_max]
                if np.mean(region) > 235:
                    for new_y in range(y, y_max):
                        if np.mean(gray[new_y, x:x_max]) < 235:
                            y = new_y
                            break
            elif edge == "bottom":
                region = gray[y_max - 10 : y_max, x:x_max]
                if np.mean(region) > 235:
                    for new_y in range(y_max - 1, y, -1):
                        if np.mean(gray[new_y, x:x_max]) < 235:
                            y_max = new_y + 1
                            break
            elif edge == "left":
                region = gray[y:y_max, x : x + 10]
                if np.mean(region) > 235:
                    for new_x in range(x, x_max):
                        if np.mean(gray[y:y_max, new_x]) < 235:
                            x = new_x
                            break
            elif edge == "right":
                region = gray[y:y_max, x_max - 10 : x_max]
                if np.mean(region) > 235:
                    for new_x in range(x_max - 1, x, -1):
                        if np.mean(gray[y:y_max, new_x]) < 235:
                            x_max = new_x + 1
                            break

        # Crop the image
        return image[y:y_max, x:x_max]

    def merge_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Merge multiple images into a single panorama
        """
        if len(images) < 2:
            raise ValueError("At least two images are required for merging")

        # Preprocess images to remove white borders
        processed_images = []
        for img in images:
            processed = self._remove_white_borders(img)
            if self.debug:
                print(
                    f"Original shape: {img.shape}, After border removal: {processed.shape}"
                )
            processed_images.append(processed)

        # Start with the first image as the base
        result = processed_images[0].copy()

        if self.debug:
            print(f"Base image shape: {result.shape}")

        for i in range(1, len(processed_images)):
            if self.debug:
                print(f"Processing image {i}, shape: {processed_images[i].shape}")

            # Try different orientations and find the best match
            kp1, oriented_img2, matches = self._try_all_orientations(
                result, processed_images[i]
            )

            if len(matches) < self.MIN_MATCHES:
                self.logger.warning(f"Not enough matches found for image {i}")
                continue

            # Compute homography
            kp2, desc2 = self._detect_and_compute(oriented_img2)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )

            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if H is None:
                raise ValueError(f"Could not find homography for image {i}")

            # Ensure H is float32
            H = H.astype(np.float32)

            # Calculate the dimensions of the merged image
            h1, w1 = result.shape[:2]
            h2, w2 = oriented_img2.shape[:2]

            # Create the panorama
            points = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(points, H)

            # Get the bounds with padding to avoid edge artifacts
            [xmin, ymin] = np.int32(transformed.min(axis=0).ravel() - 1.5)
            [xmax, ymax] = np.int32(transformed.max(axis=0).ravel() + 1.5)

            # Calculate the size of the new canvas
            x_offset = abs(min(0, xmin))
            y_offset = abs(min(0, ymin))
            new_width = max(xmax + x_offset, w1 + x_offset)
            new_height = max(ymax + y_offset, h1 + y_offset)

            if self.debug:
                print(f"Canvas dimensions: {new_height}x{new_width}")

            # Create transformation matrices
            transform = np.array(
                [
                    [1.0, 0.0, float(x_offset)],
                    [0.0, 1.0, float(y_offset)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            H_adjusted = transform.dot(H).astype(np.float32)

            # Create the final result canvas
            result_panorama = np.zeros((new_height, new_width, 3), dtype=np.uint8)

            # First, warp and copy the base image
            result_warped = cv2.warpPerspective(
                result,
                transform,
                (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),  # Use white border instead of black
            )

            # Copy base image to result
            result_panorama = result_warped.copy()

            # Then, warp the second image
            img2_warped = cv2.warpPerspective(
                oriented_img2,
                H_adjusted,
                (new_width, new_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),  # Use white border instead of black
            )

            # Create masks
            mask1 = cv2.warpPerspective(
                np.ones((h1, w1), dtype=np.uint8) * 255,
                transform,
                (new_width, new_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            mask2 = cv2.warpPerspective(
                np.ones((h2, w2), dtype=np.uint8) * 255,
                H_adjusted,
                (new_width, new_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            # Find overlap region
            overlap = cv2.bitwise_and(mask1, mask2)

            # Create alpha mask for second image
            alpha_mask = np.zeros((new_height, new_width), dtype=np.float32)
            alpha_mask[mask2 > 0] = 1.0

            # Create feathered blend in overlap region
            if overlap.any():
                # Create distance transforms
                dist1 = cv2.distanceTransform(
                    (mask1 > 0).astype(np.uint8), cv2.DIST_L2, 5
                )
                dist2 = cv2.distanceTransform(
                    (mask2 > 0).astype(np.uint8), cv2.DIST_L2, 5
                )

                # Normalize distances
                dist1 = cv2.normalize(dist1, None, 0, 1, cv2.NORM_MINMAX)
                dist2 = cv2.normalize(dist2, None, 0, 1, cv2.NORM_MINMAX)

                # Create weight mask for overlap region
                weight_mask = dist2 / (dist1 + dist2 + 1e-6)
                weight_mask = cv2.GaussianBlur(weight_mask, (15, 15), 0)

                # Update alpha mask in overlap region
                alpha_mask[overlap > 0] = weight_mask[overlap > 0]

            # Blend images using alpha mask
            for c in range(3):
                result_panorama[..., c] = (
                    result_warped[..., c] * (1 - alpha_mask)
                    + img2_warped[..., c] * alpha_mask
                ).astype(np.uint8)

            result = result_panorama

            if self.debug:
                print(f"New panorama shape: {result.shape}")

        # Final cleanup - simple crop without additional processing
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)
        x, y, w, h = cv2.boundingRect(coords)

        # Minimal border
        border = 1
        x = max(0, x - border)
        y = max(0, y - border)
        w = min(result.shape[1] - x, w + border)
        h = min(result.shape[0] - y, h + border)

        result = result[y : y + h, x : x + w]

        return result
