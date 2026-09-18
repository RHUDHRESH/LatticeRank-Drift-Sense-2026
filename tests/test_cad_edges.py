"""Focused tests for Phase 3's independent layered-edge path."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import phase3
from driftforge.cad_edges import (
    _cad_proposal_candidates,
    _correlation_ratio,
    _vector_cad_proposal_candidates,
    load_layered_template,
    solve_cad_edges,
)
from driftforge.cadref import read_gds_geometry

gdstk = pytest.importorskip("gdstk")
cv2 = pytest.importorskip("cv2")


def test_correlation_ratio_detects_cad_group_yield() -> None:
    labels = np.repeat(np.arange(4, dtype=np.uint32), 64).reshape(16, 16)
    structured = labels.astype(np.float32) * 0.25
    flat = np.ones_like(structured)
    assert _correlation_ratio(labels, structured) > 0.99
    assert _correlation_ratio(labels, flat) == 0.0


def _write_layout(path: Path, *, physical_units: bool = False) -> Path:
    factor = 0.001 if physical_units else 1.0
    library = gdstk.Library(unit=1e-6)
    cell = gdstk.Cell("REFERENCE")
    # Sparse asymmetric lower layer: this is the layer retained in SEM.
    cell.add(gdstk.rectangle((90 * factor, 120 * factor),
                             (430 * factor, 210 * factor), layer=2))
    cell.add(gdstk.rectangle((120 * factor, 210 * factor),
                             (210 * factor, 690 * factor), layer=2))
    cell.add(gdstk.rectangle((650 * factor, 700 * factor),
                             (850 * factor, 870 * factor), layer=2))
    # The highest layer dominates CAD edge length, but is deliberately absent
    # in the SEM fixture below.
    for x in range(35, 985, 35):
        cell.add(gdstk.rectangle((x * factor, 30 * factor),
                                 ((x + 8) * factor, 970 * factor), layer=11))
    library.add(cell)
    library.write_gds(str(path))
    return path


def _write_translated_layout_pair(directory: Path) -> tuple[Path, Path, tuple[int, int]]:
    """Write a local reference and its translated full-canvas CAD scene."""
    left_px, top_px = 234, 417
    offset = np.array([left_px * 10.0, top_px * 10.0])
    shapes = {
        2: [np.array(((80, 120), (430, 120), (430, 210), (80, 210))),
            np.array(((110, 210), (200, 210), (200, 720), (110, 720)))],
        7: [np.array(((610, 640), (850, 640), (850, 890), (610, 890)))],
    }

    reference_library = gdstk.Library()
    reference_cell = gdstk.Cell("REFERENCE_TRANSLATED_FIXTURE")
    search_library = gdstk.Library()
    search_cell = gdstk.Cell("SEARCH_TRANSLATED_FIXTURE")
    for layer, polygons in shapes.items():
        for points in polygons:
            reference_cell.add(gdstk.Polygon(points, layer=layer))
            search_cell.add(gdstk.Polygon(points + offset, layer=layer))
    # Pin the full scene's intended 10,000 nm frame without creating a useful
    # lookalike for either reference layer.
    search_cell.add(gdstk.rectangle((5, 5), (15, 15), layer=31))
    search_cell.add(gdstk.rectangle((9980, 9980), (9990, 9990), layer=31))
    reference_library.add(reference_cell)
    search_library.add(search_cell)
    reference_path = directory / "translated_reference.gds"
    search_path = directory / "translated_search.gds"
    reference_library.write_gds(str(reference_path))
    search_library.write_gds(str(search_path))
    return reference_path, search_path, (left_px, top_px)


def _plant_layer(gds_path: Path, left: float, top: float, *, layer: int,
                 polarity: float = 1.0, seed: int = 4) -> np.ndarray:
    template = load_layered_template(gds_path)
    visible = template.regions[template.layer_ids.index(layer)]
    canvas = np.full((520, 520), 0.45, dtype=np.float32)
    transform = np.array([[1.0, 0.0, left], [0.0, 1.0, top]], dtype=np.float32)
    planted = cv2.warpAffine(visible, transform, (520, 520), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    canvas += polarity * 0.38 * planted
    canvas += np.random.default_rng(seed).normal(0.0, 0.012, canvas.shape)
    return np.clip(canvas * 255.0, 0, 255).astype(np.uint8)


def _translate_reference_into_search_gds(reference: Path, destination: Path,
                                         left: float, top: float) -> Path:
    source = gdstk.read_gds(str(reference)).top_level()[0]
    cell = gdstk.Cell("SEARCH_FIXTURE")
    offset = np.array((left * 10.0, top * 10.0))
    for polygon in source.get_polygons():
        cell.add(gdstk.Polygon(np.asarray(polygon.points) + offset,
                               layer=polygon.layer, datatype=polygon.datatype))
    cell.add(gdstk.rectangle((5, 5), (15, 15), layer=31))
    cell.add(gdstk.rectangle((9980, 9980), (9990, 9990), layer=31))
    library = gdstk.Library()
    library.add(cell)
    library.write_gds(str(destination))
    return destination


@pytest.mark.parametrize("visible_layer", [2, 11])
@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_first_or_last_layer_can_be_invisible(tmp_path: Path, polarity: float,
                                               visible_layer: int) -> None:
    gds_path = _write_layout(tmp_path / "layers.gds")
    left, top = 173.35, 246.60
    search_gds = _translate_reference_into_search_gds(
        gds_path, tmp_path / "search.gds", left, top
    )
    result = solve_cad_edges(
        gds_path,
        _plant_layer(gds_path, left, top, layer=visible_layer,
                     polarity=polarity),
        search_gds_path=search_gds,
    )
    assert result["found"] == 1
    assert result["theta"] == 0.0 and result["scale"] == 10.0
    assert abs(float(result["x"]) - (left + 50.0)) < 1.0
    assert abs(float(result["y"]) - (top + 50.0)) < 1.0


def test_unrelated_texture_is_rejected(tmp_path: Path) -> None:
    gds_path = _write_layout(tmp_path / "layers.gds")
    noise = np.random.default_rng(91).integers(0, 256, (520, 520), dtype=np.uint8)
    result = solve_cad_edges(gds_path, noise)
    assert result["found"] == 0
    assert result["theta"] == 0.0 and result["scale"] == 0.0
    assert np.isfinite(float(result["score"]))


def test_auto_units_support_generator_and_physical_gds(tmp_path: Path) -> None:
    numeric = _write_layout(tmp_path / "numeric_nm.gds", physical_units=False)
    physical = _write_layout(tmp_path / "physical_um.gds", physical_units=True)
    numeric_geometry = read_gds_geometry(numeric)
    physical_geometry = read_gds_geometry(physical)
    assert numeric_geometry.unit_mode == "numeric-nm"
    assert numeric_geometry.nm_per_user_unit == pytest.approx(1.0)
    assert physical_geometry.unit_mode == "gds"
    assert physical_geometry.nm_per_user_unit == pytest.approx(1000.0)
    assert numeric_geometry.origin_nm == physical_geometry.origin_nm == (0.0, 0.0)


def test_search_gds_and_training_metadata_are_never_reference_fallback() -> None:
    row = {
        "search_gds_path": "search/full.gds",
        "reference_sem_path": "reference/training.png",
        "params_json_path": "parameters/answer.json",
        "search_path": "search/image.png",
    }
    assert phase3._pick_gds(row) is None
    assert phase3._pick_search_gds(row) == "search/full.gds"
    row["unlabelled_reference"] = "reference/crop.gds"
    assert phase3._pick_gds(row) == "reference/crop.gds"


def test_training_metadata_is_not_a_search_gds_fallback() -> None:
    assert phase3._pick_search_gds({
        "reference_sem_path": "reference/training.png",
        "params_json_path": "parameters/answer.json",
        "unlabelled_gds": "answers/ground_truth.gds",
    }) is None


def test_full_canvas_cad_proposal_recovers_translation_and_numeric_nm_units(
        tmp_path: Path) -> None:
    reference, search_gds, expected = _write_translated_layout_pair(tmp_path)
    template = load_layered_template(reference)
    candidates = _cad_proposal_candidates(
        search_gds, template, (1000, 1000), reference_gds_path=reference
    )
    assert candidates
    row, col, score = candidates[0]
    assert (col, row) == pytest.approx(expected, abs=1)
    assert score > 0.95
    assert read_gds_geometry(search_gds, window_nm=10_000).unit_mode == "numeric-nm"


def test_vector_coverage_uses_all_reference_polygons(tmp_path: Path) -> None:
    reference_library = gdstk.Library()
    reference_cell = gdstk.Cell("REFERENCE_COVERAGE")
    search_library = gdstk.Library()
    search_cell = gdstk.Cell("SEARCH_COVERAGE")

    for index in range(100):
        width = 10.0 + index
        points = np.array(((0.0, 0.0), (width, 0.0),
                           (width, 12.0), (0.0, 12.0)))
        reference_cell.add(gdstk.Polygon(points + (50.0, index * 8.0),
                                         layer=2))
        if index == 0:
            # Duplicate search polygons at the same centroid must still
            # explain only one reference polygon, not inflate coverage.
            for _duplicate in range(10):
                search_cell.add(gdstk.Polygon(points + (2050.0, 3000.0),
                                              layer=2))

    search_cell.add(gdstk.rectangle((5, 5), (15, 15), layer=31))
    search_cell.add(gdstk.rectangle((9980, 9980), (9990, 9990), layer=31))
    reference_library.add(reference_cell)
    search_library.add(search_cell)
    reference_path = tmp_path / "coverage_reference.gds"
    search_path = tmp_path / "coverage_search.gds"
    reference_library.write_gds(str(reference_path))
    search_library.write_gds(str(search_path))

    candidates = _vector_cad_proposal_candidates(
        reference_path, search_path, (1000, 1000), 10.0
    )
    assert candidates
    assert candidates[0][2] == pytest.approx(0.01)
    assert candidates[0][2] < 0.04


def _write_dense_vector_pair(directory: Path, *, reverse: bool) -> tuple[Path, Path]:
    """A repeated signature exceeds the pair cap; one rare shape is last."""
    offset = np.array((2400.0, 3600.0))
    common = []
    for index in range(800):
        x = 20.0 + (index % 40) * 20.0
        y = 20.0 + (index // 40) * 20.0
        common.append(np.array(((x, y), (x + 6.0, y),
                                (x + 6.0, y + 4.0), (x, y + 4.0))))
    rare = np.array(((873.0, 811.0), (901.0, 817.0), (882.0, 849.0)))
    polygons = [(2, points) for points in common] + [(9, rare)]
    if reverse:
        polygons.reverse()

    reference_library = gdstk.Library()
    reference_cell = gdstk.Cell(f"DENSE_REFERENCE_{int(reverse)}")
    search_library = gdstk.Library()
    search_cell = gdstk.Cell(f"DENSE_SEARCH_{int(reverse)}")
    for layer, points in polygons:
        reference_cell.add(gdstk.Polygon(points, layer=layer))
        search_cell.add(gdstk.Polygon(points + offset, layer=layer))
    reference_library.add(reference_cell)
    search_library.add(search_cell)
    reference_path = directory / f"dense_reference_{int(reverse)}.gds"
    search_path = directory / f"dense_search_{int(reverse)}.gds"
    reference_library.write_gds(str(reference_path))
    search_library.write_gds(str(search_path))
    return reference_path, search_path


def test_vector_votes_are_order_invariant_beyond_pair_cap(tmp_path: Path) -> None:
    forward = _write_dense_vector_pair(tmp_path, reverse=False)
    reversed_pair = _write_dense_vector_pair(tmp_path, reverse=True)

    forward_candidates = _vector_cad_proposal_candidates(
        *forward, (1000, 1000), 10.0
    )
    reversed_candidates = _vector_cad_proposal_candidates(
        *reversed_pair, (1000, 1000), 10.0
    )

    assert forward_candidates
    assert reversed_candidates
    assert forward_candidates[0][:2] == (360, 240)
    assert reversed_candidates[0][:2] == (360, 240)
    assert forward_candidates[0][2] == pytest.approx(1.0)
    assert reversed_candidates[0] == pytest.approx(forward_candidates[0])


def test_cli_routes_search_gds_without_reading_training_json(
        tmp_path: Path, monkeypatch) -> None:
    reference, search_gds, expected = _write_translated_layout_pair(tmp_path)
    search_image = tmp_path / "search.png"
    assert cv2.imwrite(str(search_image), np.zeros((1000, 1000), dtype=np.uint8))
    poison = tmp_path / "training-only.json"
    poison.write_text("this is deliberately not JSON", encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    with pairs.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "pair_id", "reference_gds_path", "search_path",
            "search_gds_path", "params_json_path",
        ))
        writer.writeheader()
        writer.writerow({
            "pair_id": "legal-inputs",
            "reference_gds_path": reference.name,
            "search_path": search_image.name,
            "search_gds_path": search_gds.name,
            "params_json_path": poison.name,
        })

    observed = {}

    def fake_solver(gds_path, image, *, scale, search_gds_path=None):
        observed.update(reference=Path(gds_path), search_gds=Path(search_gds_path),
                        shape=image.shape, scale=scale)
        return {"x": expected[0] + 50.0, "y": expected[1] + 50.0,
                "theta": 0.0, "scale": 10.0, "found": 1, "score": 0.9}

    monkeypatch.setattr(phase3, "solve_cad_edges", fake_solver)
    output = tmp_path / "predictions.csv"
    assert phase3.main(["--input", str(pairs), "--output", str(output)]) == 0
    assert observed == {
        "reference": reference.resolve(),
        "search_gds": search_gds.resolve(),
        "shape": (1000, 1000),
        "scale": 10.0,
    }
    assert poison.read_text(encoding="utf-8") == "this is deliberately not JSON"


def test_layer_limit_rejects_instead_of_silently_truncating(tmp_path: Path) -> None:
    library = gdstk.Library()
    cell = gdstk.Cell("TOO_MANY_LAYERS")
    for layer in range(33):
        offset = 10 + layer * 2
        cell.add(gdstk.rectangle((offset, offset), (offset + 20, offset + 20),
                                 layer=layer))
    library.add(cell)
    path = tmp_path / "too_many.gds"
    library.write_gds(str(path))
    with pytest.raises(ValueError, match="33 layers.*limit is 32"):
        load_layered_template(path)


def test_opencv_thread_budget_is_restored(tmp_path: Path, monkeypatch) -> None:
    gds_path = _write_layout(tmp_path / "layers.gds")
    search = _plant_layer(gds_path, 80.0, 110.0, layer=2)
    previous = int(cv2.getNumThreads())
    sentinel = 2 if previous != 2 else 3
    cv2.setNumThreads(sentinel)
    monkeypatch.setenv("DRIFTFORGE_NUM_THREADS", "1")
    try:
        solve_cad_edges(gds_path, search)
        assert cv2.getNumThreads() == sentinel
    finally:
        cv2.setNumThreads(previous)
