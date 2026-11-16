import json

from src.annotation_comparator.comparison import compare
from src.annotation_comparator.parser import load_annotation


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


def test_different_types_do_not_match(tmp_path):
    result = compare(annotation(tmp_path, "a", [("a", "car", (0, 0, 5, 5))]),
                     annotation(tmp_path, "g", [("g", "road sign", (0, 0, 5, 5))]))
    assert result.score == 0.0
