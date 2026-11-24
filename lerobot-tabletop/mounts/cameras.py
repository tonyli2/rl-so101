"""Native MuJoCo pinhole intrinsics matching OpenCV's undistorted pixel frame."""

import numpy as np


def apply_intrinsics(camera, measured):
    """Use measured K at its native resolution; no image warp in the sim loop.

    Real frames must be undistorted using the same K as the output matrix.
    MuJoCo principalpixel is a signed frustum offset, not OpenCV's absolute cx/cy.
    The virtual sensor dimensions set a unit scale only: focalpixel determines
    the measured projection without assuming the hardware sensor's physical size.
    """
    matrix = np.asarray(measured["camera_matrix"], dtype=float)
    width, height = int(measured["width"]), int(measured["height"])
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or min(width, height) <= 0
        or min(matrix[0, 0], matrix[1, 1]) <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
    ):
        raise ValueError("Expected finite zero-skew pinhole K and positive image size")
    camera.attrib.pop("fovy", None)
    camera.set("resolution", f"{width} {height}")
    camera.set("sensorsize", "0.0064 0.0048")
    camera.set("focalpixel", f"{matrix[0, 0]:.12g} {matrix[1, 1]:.12g}")
    camera.set(
        "principalpixel",
        f"{width / 2 - matrix[0, 2]:.12g} {height / 2 - matrix[1, 2]:.12g}",
    )
