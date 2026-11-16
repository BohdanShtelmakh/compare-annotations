"""Loading and normalization of annotation JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AnnotationError(ValueError):
    """Raised when an annotation file does not have the expected structure."""


@dataclass(frozen=True)
class AnnotationObject:
    nm: str
    type: str | None
    shape: str | None
    bbox: tuple[float, float, float, float]
    path: tuple[tuple[float, float], ...] | None = None
    mask_path: Path | None = None


@dataclass(frozen=True)
class Annotation:
    file: str
    width: int
    height: int
    objects: tuple[AnnotationObject, ...]


def load_annotation(json_path: str | Path, masks_dir: str | Path | None = None) -> Annotation:
    """Load one annotation export and link ``nm``-named masks when available."""
    path = Path(json_path)
    try:
        pages = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"cannot read annotation file {path}: {exc}") from exc

    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        raise AnnotationError("annotation JSON must be a non-empty list")
    header = pages[0]
    objects_by_nm = {obj["nm"]: obj for obj in _objects(header, "header")}
    if len(objects_by_nm) != len(header.get("objects", [])):
        raise AnnotationError("duplicate object nm in header")

    geometry_objects = _objects(pages[1], "geometry") if len(pages) > 1 else []
    result = []
    seen = set()
    for raw in geometry_objects:
        nm = raw["nm"]
        if nm in seen:
            raise AnnotationError(f"duplicate object nm in geometry: {nm}")
        seen.add(nm)
        if nm not in objects_by_nm:
            raise AnnotationError(f"unknown object nm in geometry: {nm}")
        definition = objects_by_nm[nm]
        result.append(_normalize_object({**definition, **raw}, masks_dir))

    missing = set(objects_by_nm) - seen
    if missing:
        raise AnnotationError(f"objects missing geometry: {sorted(missing)}")

    try:
        width, height = int(header["width"]), int(header["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationError("header must contain integer width and height") from exc
    if width <= 0 or height <= 0:
        raise AnnotationError("width and height must be positive")
    return Annotation(str(header.get("file", path.name)), width, height, tuple(result))


def _objects(page: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(page, dict) or not isinstance(page.get("objects"), list):
        raise AnnotationError(f"{label} page must contain an objects list")
    if not all(isinstance(obj, dict) and isinstance(obj.get("nm"), str) and obj["nm"] for obj in page["objects"]):
        raise AnnotationError(f"each {label} object must have a non-empty string nm")
    return page["objects"]


def _normalize_object(raw: dict[str, Any], masks_dir: str | Path | None) -> AnnotationObject:
    try:
        values = tuple(float(raw[key]) for key in ("x1", "y1", "x2", "y2"))
        bbox: tuple[float, float, float, float] = (values[0], values[1], values[2], values[3])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnnotationError(f"object {raw.get('nm')} has an invalid bounding box") from exc
    if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
        raise AnnotationError(f"object {raw['nm']} has inverted bounding box")

    polygon = raw.get("path")
    normalized_path = None
    if polygon is not None:
        if len(polygon) == 1 and isinstance(polygon[0], list):
            polygon = polygon[0]
        try:
            normalized_path = tuple((float(point[0]), float(point[1])) for point in polygon)
        except (TypeError, IndexError, ValueError) as exc:
            raise AnnotationError(f"object {raw['nm']} has an invalid polygon path") from exc

    mask_path = None
    if masks_dir is not None:
        candidate = Path(masks_dir) / f"{raw['nm']}.png"
        if not candidate.exists():
            raise AnnotationError(f"missing mask for object {raw['nm']}: {candidate}")
        mask_path = candidate
    return AnnotationObject(raw["nm"], raw.get("type"), raw.get("shape"), bbox, normalized_path, mask_path)
