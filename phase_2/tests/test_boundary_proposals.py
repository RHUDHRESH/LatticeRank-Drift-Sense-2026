import cv2
import numpy as np

from driftforge.boundary_proposals import _render, boundary_pose_proposals


def test_constant_boundary_proposal_recovers_rotated_target() -> None:
    rng = np.random.default_rng(31)
    reference = cv2.GaussianBlur(
        rng.normal(120, 45, (240, 240)).astype(np.float32), (0, 0), 2.0)
    cv2.rectangle(reference, (31, 47), (92, 121), 245, 9)
    template = _render(reference, 12.0, 10.0, cv2.BORDER_CONSTANT)
    search = rng.normal(90, 7, (220, 230)).astype(np.float32)
    row, col = 83, 117
    search[row:row+template.shape[0], col:col+template.shape[1]] = 255-template

    proposals = boundary_pose_proposals(
        reference, search, scales=(12.0,), angles=(10.0,), shortlist=2)

    expected_x = col + (template.shape[1]-1.0)/2.0
    expected_y = row + (template.shape[0]-1.0)/2.0
    assert proposals
    assert min(np.hypot(item.x-expected_x, item.y-expected_y)
               for item in proposals) <= 0.1


def test_invalid_flat_reference_has_no_proposals() -> None:
    assert boundary_pose_proposals(
        np.zeros((80, 80), np.float32), np.zeros((160, 160), np.float32)) == []
