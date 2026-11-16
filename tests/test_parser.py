import json

import pytest

from src.annotation_comparator.parser import AnnotationError, load_annotation


def write_annotation(tmp_path, pages):
    path = tmp_path / "annotation.json"
    path.write_text(json.dumps(pages), encoding="utf-8")
    return path


def pages():
    return [
        {"file": "road.jpg", "width": 100, "height": 80, "objects": [
            {"nm": "0.1", "type": "car", "shape": "poly"},
        ]},
        {"objects": [{"nm": "0.1", "x1": 1, "y1": 2, "x2": 10, "y2": 20,
                      "path": [[[1, 2], [10, 2], [10, 20]]]}]},
    ]


def test_loads_and_normalizes_annotation(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "0.1.png").touch()

    annotation = load_annotation(write_annotation(tmp_path, pages()), masks)

    assert annotation.file == "road.jpg"
    assert (annotation.width, annotation.height) == (100, 80)
    obj = annotation.objects[0]
    assert obj.nm == "0.1"
    assert obj.bbox == (1.0, 2.0, 10.0, 20.0)
    assert obj.path == ((1.0, 2.0), (10.0, 2.0), (10.0, 20.0))
    assert obj.mask_path == masks / "0.1.png"


def test_unknown_geometry_id_is_rejected(tmp_path):
    data = pages()
    data[1]["objects"][0]["nm"] = "unknown"
    with pytest.raises(AnnotationError, match="unknown object nm"):
        load_annotation(write_annotation(tmp_path, data))


def test_missing_geometry_is_rejected(tmp_path):
    data = pages()
    data[1]["objects"] = []
    with pytest.raises(AnnotationError, match="missing geometry"):
        load_annotation(write_annotation(tmp_path, data))


def test_missing_required_mask_is_rejected(tmp_path):
    with pytest.raises(AnnotationError, match="missing mask"):
        load_annotation(write_annotation(tmp_path, pages()), tmp_path / "masks")


def test_invalid_bbox_is_rejected(tmp_path):
    data = pages()
    data[1]["objects"][0]["x2"] = 0
    with pytest.raises(AnnotationError, match="inverted bounding box"):
        load_annotation(write_annotation(tmp_path, data))
