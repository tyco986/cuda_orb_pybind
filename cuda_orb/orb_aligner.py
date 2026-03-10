"""
ORB CUDA aligner: GPU-accelerated ORB features + match + RANSAC homography.
"""

import cv2
import numpy as np
from typing import Dict, Tuple

from ._cuda_orb import OrbPipeline, init_device

DRATIO = 0.75


def _to_gray_u8(x) -> np.ndarray:
    """(B,H,W) or (B,C,H,W) -> (B,H,W) uint8 [0,255]. Accepts numpy or array-like."""
    arr = np.asarray(x)
    if arr.ndim == 4:
        if arr.shape[1] == 3:
            arr = arr[:, 0] * 0.299 + arr[:, 1] * 0.587 + arr[:, 2] * 0.114
        else:
            arr = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 3:
            arr = np.dot(arr.transpose(1, 2, 0), [0.299, 0.587, 0.114])
        else:
            arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.dtype == np.uint8:
        return arr
    alpha = 255.0 if arr.flat[0] <= 1.0 else 1.0
    out = np.empty(arr.shape, dtype=np.uint8)
    for b in range(arr.shape[0]):
        cv2.convertScaleAbs(arr[b], dst=out[b], alpha=alpha)
    return out


class OrbAligner:
    """
    ORB CUDA: GPU-accelerated ORB features + Hamming match + RANSAC homography.

    Uses OrbPipeline (detect + binary descriptor + brute-force Hamming match)
    and cv2.findHomography (RANSAC) to compute homography between image pairs.
    """

    def __init__(
        self,
        ransac_thresh: float = 5.0,
        ransac_max_iters: int = 200,
        ransac_confidence: float = 0.99,
        ransac_seed: int | None = None,
        nndr: float = DRATIO,
        max_pts: int = 10000,
        noctaves: int = 5,
        fast_threshold: int = 20,
        device: int = 0,
        use_nndr: bool = True,
    ):
        self.ransac_thresh = ransac_thresh
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.ransac_seed = ransac_seed
        self.nndr = nndr if use_nndr else 1.0
        self.device = device
        init_device(device)
        self._pipeline = OrbPipeline(
            max_pts=max_pts,
            noctaves=noctaves,
            fast_threshold=fast_threshold,
        )

    def _find_transform_one(
        self, template: np.ndarray, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single pair. Returns (H 3x3, motion 2,)."""
        pts0, pts1 = self._pipeline.detect_and_match(
            template, image, self.nndr,
        )

        if pts0.shape[0] < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        if self.ransac_seed is not None:
            cv2.setRNGSeed(self.ransac_seed)

        H, mask = cv2.findHomography(
            pts0, pts1, cv2.RANSAC,
            self.ransac_thresh,
            maxIters=self.ransac_max_iters,
            confidence=self.ransac_confidence,
        )
        if H is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        return H.astype(np.float32), H[:2, 2].astype(np.float32)

    def find_transform(self, template, input_image) -> Dict:
        """
        template, input_image: (B,H,W) or (B,C,H,W), numpy or array-like.
        Returns: {"warp_matrix": (B,3,3), "motion": (B,2)}
        """
        t = _to_gray_u8(template)
        i = _to_gray_u8(input_image)
        if t.ndim == 2:
            t = t[np.newaxis]
            i = i[np.newaxis]

        B = t.shape[0]
        warps = []
        motions = []
        for b in range(B):
            H, mot = self._find_transform_one(t[b], i[b])
            warps.append(H)
            motions.append(mot)

        return {
            "warp_matrix": np.stack(warps, axis=0),
            "motion": np.stack(motions, axis=0),
        }


__all__ = ["OrbAligner", "DRATIO"]
