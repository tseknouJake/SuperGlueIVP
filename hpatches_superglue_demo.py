import argparse
from pathlib import Path

import cv2
import torch
import numpy as np

from models.matching import Matching
from models.utils import (read_image, frame2tensor)


def run_on_pair(img0_path, img1_path,
                resize=640,
                superglue_weights='outdoor',
                max_keypoints=1024,
                keypoint_threshold=0.005,
                nms_radius=4,
                match_threshold=0.2,
                device='cuda' if torch.cuda.is_available() else 'cpu'):

    print(f"Using device: {device}")
    device = torch.device(device)

    # Configuration copied from match_pairs.py, adapted for HPatches
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints,
        },
        'superglue': {
            'weights': superglue_weights,  # 'indoor' or 'outdoor'
            'sinkhorn_iterations': 20,
            'match_threshold': match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)

    # HPatches images don’t need EXIF rotation; set to 0
    # resize: 640 means "max side 640", like in the README
    print("Loading images...")
    image0, inp0, _ = read_image(
        str(img0_path), device, resize, rotation=0, resize_float=False
    )
    image1, inp1, _ = read_image(
        str(img1_path), device, resize, rotation=0, resize_float=False
    )

    # Run SuperPoint + SuperGlue
    print("Running SuperGlue...")
    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})

    # Extract keypoints and matches
    kpts0 = pred['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches0 = pred['matches0'][0].cpu().numpy()
    conf0 = pred['matching_scores0'][0].cpu().numpy()

    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]
    mconf = conf0[valid]

    print(f"# keypoints image0: {len(kpts0)}")
    print(f"# keypoints image1: {len(kpts1)}")
    print(f"# matches: {len(mkpts0)} (after filtering)")

    # Optionally, estimate a homography directly from these matches (RANSAC)
    if len(mkpts0) >= 4:
        H, inliers = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0)
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        print(f"Estimated H with {inlier_count}/{len(mkpts0)} inliers")
    else:
        H, inliers = None, None
        print("Not enough matches for homography.")

    return {
        'kpts0': kpts0,
        'kpts1': kpts1,
        'mkpts0': mkpts0,
        'mkpts1': mkpts1,
        'mconf': mconf,
        'H': H,
        'inliers': inliers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run SuperGlue on one HPatches pair."
    )
    parser.add_argument('img0', type=str,
                        help='Path to first HPatches image (e.g. 1.ppm)')
    parser.add_argument('img1', type=str,
                        help='Path to second HPatches image (e.g. 2.ppm)')
    parser.add_argument('--resize', type=int, default=640,
                        help='Resize max dimension (default: 640, -1 = no resize)')
    parser.add_argument('--superglue', type=str, default='outdoor',
                        choices=['indoor', 'outdoor'],
                        help='SuperGlue weights to use')
    parser.add_argument('--max_keypoints', type=int, default=1024)
    parser.add_argument('--keypoint_threshold', type=float, default=0.005)
    parser.add_argument('--nms_radius', type=int, default=4)
    parser.add_argument('--match_threshold', type=float, default=0.2)
    args = parser.parse_args()

    img0_path = Path(args.img0)
    img1_path = Path(args.img1)
    assert img0_path.exists(), f"Image not found: {img0_path}"
    assert img1_path.exists(), f"Image not found: {img1_path}"

    run_on_pair(
        img0_path, img1_path,
        resize=args.resize,
        superglue_weights=args.superglue,
        max_keypoints=args.max_keypoints,
        keypoint_threshold=args.keypoint_threshold,
        nms_radius=args.nms_radius,
        match_threshold=args.match_threshold,
    )


if __name__ == '__main__':
    main()
