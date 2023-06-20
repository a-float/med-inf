from pathlib import Path
from dataclasses import dataclass
import heapq

import cv2

@dataclass
class FileMatch:
    name: str
    image: cv2.Mat
    keypoints: tuple[cv2.KeyPoint, ...]
    matchpoints: list[cv2.DMatch]


test_original = cv2.imread("Altered-custom/1__M_Left_index_finger_rot_czesc.BMP")
data_dir = Path("Real_subset")
orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create()


def preprocess(image: cv2.Mat) -> cv2.Mat:
    # todo preprocessing
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def extract_features(image: cv2.Mat):
    # todo feature extraction - sift, surf, fast, brief ...
    return sift.detectAndCompute(image, None)
    # return orb.detectAndCompute(image, None)


test_preprocessed = preprocess(test_original)
test_keypoints, test_descriptors = extract_features(test_preprocessed)

# print(np.min(test_preprocessed), np.max(test_preprocessed))
# cv2.imshow("Original", cv2.resize(test_preprocessed, None, fx=1, fy=1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

results: list[FileMatch] = []

for file in data_dir.iterdir():
    fingerprint_image = cv2.imread(str(file))

    keypoints, descriptors = extract_features(
        preprocess(fingerprint_image)
    )

    matches = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=100)
    ).knnMatch(test_descriptors, descriptors, k=2)
    # alternative matcher that can be used with orb descriptor
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = [
        p for p, q in matches if p.distance < 0.7 * q.distance
    ]

    max_keypoints = max(len(test_keypoints), len(keypoints))
    
    if (len(match_points) / max_keypoints) > 0:
        results.append(
            FileMatch(
                name=file.name,
                image=fingerprint_image,
                keypoints=keypoints,
                matchpoints=match_points,
            )
        )

final_results = heapq.nlargest(5, results, key=lambda x: len(x.matchpoints))

for file_match in final_results:
    kp = max(len(file_match.keypoints), len(test_keypoints))
    percent = len(file_match.matchpoints) / kp * 100

    print(f"Fingerprint ID: {file_match.name}")
    print(f"\t{percent:.2f}% match: {len(file_match.matchpoints)} out of {kp} points")

    result = cv2.drawMatches(
        test_original,
        test_keypoints,
        file_match.image,
        file_match.keypoints,
        file_match.matchpoints,
        None,
    )
    result = cv2.resize(result, None, fx=2.5, fy=2.5)
    
    cv2.imshow(file_match.name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
