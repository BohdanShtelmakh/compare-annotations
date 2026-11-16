"""Geometry-based annotation comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from .parser import Annotation, AnnotationObject


@dataclass(frozen=True)
class ObjectScore:
    annotated_nm: str | None
    ground_truth_nm: str | None
    object_type: str | None
    intersection: int
    union: int
    iou: float


@dataclass(frozen=True)
class ComparisonResult:
    objects: tuple[ObjectScore, ...]
    intersection: int
    union: int

    @property
    def score(self) -> float:
        return self.intersection / self.union * 100 if self.union else 100.0


def compare(annotated: Annotation, ground_truth: Annotation) -> ComparisonResult:
    if (annotated.width, annotated.height) != (ground_truth.width, ground_truth.height):
        raise ValueError("annotations must have the same dimensions")
    annotated_masks = [_to_mask(obj, annotated.width, annotated.height) for obj in annotated.objects]
    truth_masks = [_to_mask(obj, ground_truth.width, ground_truth.height) for obj in ground_truth.objects]
    pairs: list[ObjectScore] = []
    matched_a: set[int] = set()
    matched_g: set[int] = set()
    types = {obj.type for obj in annotated.objects} | {obj.type for obj in ground_truth.objects}

    for object_type in types:
        a_indices = [i for i, obj in enumerate(annotated.objects) if obj.type == object_type]
        g_indices = [i for i, obj in enumerate(ground_truth.objects) if obj.type == object_type]
        if not a_indices or not g_indices:
            continue
        intersections = np.zeros((len(a_indices), len(g_indices)), dtype=int)
        ious = np.zeros((len(a_indices), len(g_indices)))
        geometry: dict[tuple[int, int], tuple[int, int]] = {}
        for row, a_index in enumerate(a_indices):
            for column, g_index in enumerate(g_indices):
                if not _boxes_overlap(annotated.objects[a_index].bbox, ground_truth.objects[g_index].bbox):
                    continue
                a_mask, g_mask = annotated_masks[a_index], truth_masks[g_index]
                intersection = int(np.logical_and(a_mask, g_mask).sum())
                if intersection:
                    union = int(np.logical_or(a_mask, g_mask).sum())
                    intersections[row, column] = intersection
                    ious[row, column] = intersection / union
                    geometry[(row, column)] = (intersection, union)
        rows, columns = linear_sum_assignment(-intersections)
        for row, column in zip(rows, columns, strict=True):
            if ious[row, column] == 0:
                continue
            a_index, g_index = a_indices[row], g_indices[column]
            intersection, union = geometry[(row, column)]
            matched_a.add(a_index)
            matched_g.add(g_index)
            pairs.append(ObjectScore(annotated.objects[a_index].nm, ground_truth.objects[g_index].nm,
                                     object_type, intersection, union, float(ious[row, column])))

    for index, (obj, mask) in enumerate(zip(annotated.objects, annotated_masks, strict=True)):
        if index not in matched_a:
            pairs.append(ObjectScore(obj.nm, None, obj.type, 0, int(mask.sum()), 0.0))
    for index, (obj, mask) in enumerate(zip(ground_truth.objects, truth_masks, strict=True)):
        if index not in matched_g:
            pairs.append(ObjectScore(None, obj.nm, obj.type, 0, int(mask.sum()), 0.0))

    intersection = sum(item.intersection for item in pairs)
    union = sum(item.union for item in pairs)
    return ComparisonResult(tuple(pairs), intersection, union)


def _boxes_overlap(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    return not (first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1])


def _to_mask(obj: AnnotationObject, width: int, height: int) -> np.ndarray:
    if obj.mask_path is not None:
        image = Image.open(obj.mask_path)
        if image.size != (width, height):
            raise ValueError(f"mask {obj.mask_path} has incorrect dimensions")
        values = np.asarray(image)
        return values != 0 if image.mode in {"1", "P"} else np.asarray(image.convert("L")) > 0

    if obj.shape == "poly" and not obj.path:
        raise ValueError(f"polygon object {obj.nm} has no polygon path or mask")
    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)
    if obj.path:
        draw.polygon(obj.path, fill=1)
    else:
        x1, y1, x2, y2 = obj.bbox
        draw.rectangle((x1, y1, x2, y2), fill=1)
    return np.asarray(image, dtype=bool)
