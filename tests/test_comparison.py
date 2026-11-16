import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.annotation_comparator.comparison import compare
from src.annotation_comparator.parser import (
    Annotation,
    AnnotationObject,
    load_annotation,
)


def annotation(tmp_path, name, objects):
    data = [{"file": name, "width": 20, "height": 20,
             "objects": [{"nm": nm, "type": kind, "shape": "bBox"} for nm, kind, *_ in objects]},
            {"objects": [{"nm": nm, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
                          for nm, _, box in objects]}]
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_annotation(path)


def test_perfect_overlap_matches_without_using_nm(tmp_path):
    result = compare(annotation(tmp_path, "a", [("a", "car", (1, 1, 5, 5))]),
                     annotation(tmp_path, "g", [("different", "car", (1, 1, 5, 5))]))
    assert result.objects[0].iou == 1.0
    assert result.score == 100.0


def test_partial_overlap_uses_geometry(tmp_path):
    result = compare(annotation(tmp_path, "a", [("a", "car", (0, 0, 10, 10))]),
                     annotation(tmp_path, "g", [("g", "car", (5, 0, 15, 10))]))
    assert result.objects[0].intersection == 66
    assert result.objects[0].union == 176
    assert round(result.score, 2) == 37.5


def test_missing_and_extra_objects_use_their_areas(tmp_path):
    result = compare(annotation(tmp_path, "a", [("a", "car", (0, 0, 5, 5))]),
                     annotation(tmp_path, "g", [("g", "car", (0, 0, 5, 5)),
                                                  ("extra", "car", (10, 10, 15, 15))]))
    assert len(result.objects) == 2
    assert result.score == 50.0


def test_matching_maximizes_intersection_not_iou(tmp_path):
    def obj(nm, mask_name):
        return AnnotationObject(nm, "car", "poly", (0, 0, 7, 0), mask_path=tmp_path / mask_name)

    masks = {
        "a1.png": [1, 0, 0, 0, 0, 0, 0, 0],
        "a2.png": [1, 1, 1, 1, 0, 0, 0, 0],
        "g1.png": [1, 1, 1, 1, 1, 1, 1, 1],
        "g2.png": [1, 1, 1, 0, 0, 0, 0, 0],
    }
    for name, values in masks.items():
        Image.fromarray(np.array([values], dtype=np.uint8), mode="P").save(tmp_path / name)
    annotated = Annotation("a", 8, 1, (obj("a1", "a1.png"), obj("a2", "a2.png")))
    truth = Annotation("g", 8, 1, (obj("g1", "g1.png"), obj("g2", "g2.png")))

    result = compare(annotated, truth)

    assert {(item.annotated_nm, item.ground_truth_nm) for item in result.objects} == {("a1", "g2"), ("a2", "g1")}
    assert result.intersection == 5


def test_polygon_and_png_mask_compare_equal(tmp_path):
    polygon = AnnotationObject("polygon", "car", "poly", (0, 0, 3, 3),
                               path=((0, 0), (3, 0), (0, 3)))
    mask_path = tmp_path / "triangle.png"
    image = Image.new("1", (4, 4))
    ImageDraw.Draw(image).polygon(((0, 0), (3, 0), (0, 3)), fill=1)
    image.save(mask_path)
    mask = AnnotationObject("mask", "car", "poly", (0, 0, 3, 3), mask_path=mask_path)

    result = compare(Annotation("a", 4, 4, (polygon,)), Annotation("g", 4, 4, (mask,)))

    assert result.score == 100.0


def test_empty_annotations_score_perfectly(tmp_path):
    empty = Annotation("empty", 4, 4, ())
    result = compare(empty, empty)
    assert result.score == 100.0
    assert result.intersection == result.union == 0


def test_no_overlap_scores_zero(tmp_path):
    result = compare(annotation(tmp_path, "a", [("a", "car", (0, 0, 1, 1))]),
                     annotation(tmp_path, "g", [("g", "car", (10, 10, 11, 11))]))
    assert result.score == 0.0


def test_score_is_area_weighted(tmp_path):
    result = compare(annotation(tmp_path, "a", [("large", "car", (0, 0, 10, 10)),
                                                  ("small", "car", (15, 15, 16, 16))]),
                     annotation(tmp_path, "g", [("large-g", "car", (0, 0, 10, 10)),
                                                  ("small-g", "car", (15, 15, 17, 17))]))
    assert result.score == 125 / 130 * 100


def test_palette_mask_uses_nonzero_indices(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="P").save(path)
    obj = AnnotationObject("a", "car", "poly", (0, 0, 1, 0), mask_path=path)
    annotation = Annotation("a", 2, 1, (obj,))

    result = compare(annotation, annotation)

    assert result.objects[0].intersection == 1
    assert result.score == 100.0


def test_polygon_without_path_or_mask_is_rejected(tmp_path):
    obj = AnnotationObject("a", "car", "poly", (0, 0, 1, 1))
    annotation = Annotation("a", 2, 2, (obj,))

    with pytest.raises(ValueError, match="no polygon path or mask"):
        compare(annotation, annotation)


def test_different_types_do_not_match(tmp_path):
    result = compare(annotation(tmp_path, "a", [("a", "car", (0, 0, 5, 5))]),
                     annotation(tmp_path, "g", [("g", "road sign", (0, 0, 5, 5))]))
    assert result.score == 0.0
