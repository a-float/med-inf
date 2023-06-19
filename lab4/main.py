import cv2
import os
import numpy as np
from collections import namedtuple

FileMatch = namedtuple("FileMatch", ["name", "image", "keypoints", "matchpoints"])
test_original = cv2.imread("Altered-custom/1__M_Left_index_finger_rot_czesc.BMP")
dataDir = "Real_subset"
orb = cv2.ORB_create()
sift = cv2.xfeatures2d.SIFT_create()


def preprocess(image):
    # todo preprocessing
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def features_extraction(image):
    # todo feature extraction - sift, surf, fast, brief ...
    return sift.detectAndCompute(image, None)
    # return orb.detectAndCompute(image, None)


test_preprocessed = preprocess(test_original)
# print(np.min(test_preprocessed), np.max(test_preprocessed))
# cv2.imshow("Original", cv2.resize(test_preprocessed, None, fx=1, fy=1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

keypoints_1, descriptors_1 = features_extraction(test_preprocessed)

results: list[FileMatch] = []
for file in [file for file in os.listdir("Real_subset")]:
    fingerprint_database_image = cv2.imread("./Real_subset/" + file)

    keypoints_2, descriptors_2 = features_extraction(
        preprocess(fingerprint_database_image)
    )

    matches = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=100)
    ).knnMatch(descriptors_1, descriptors_2, k=2)
    # alternative matcher that can be used with orb descriptor
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

    match_points = []
    for p, q in matches:
        if p.distance < 0.7 * q.distance:
            match_points.append(p)

    keypoints = max(len(keypoints_1), len(keypoints_2))
    if (len(match_points) / keypoints) > 0:
        results.append(
            FileMatch(
                name=file,
                image=fingerprint_database_image,
                keypoints=keypoints_2,
                matchpoints=match_points,
            )
        )

results.sort(key=lambda x: -len(x.matchpoints))

for file_match in results[:5]:
    print("Fingerprint ID: " + str(file_match.name))
    kp = max(len(file_match.keypoints), len(keypoints_1))
    percent = len(file_match.matchpoints) / kp * 100
    print(f"\t{percent:.2f}% match: {len(file_match.matchpoints)} out of {kp} points")
    result = cv2.drawMatches(
        test_original,
        keypoints_1,
        file_match.image,
        file_match.keypoints,
        file_match.matchpoints,
        None,
    )
    result = cv2.resize(result, None, fx=2.5, fy=2.5)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
