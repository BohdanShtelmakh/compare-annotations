"""Geometry-based annotation comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

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
    used_truth: set[int] = set()

    for a_obj, a_mask in zip(annotated.objects, annotated_masks, strict=True):
        candidates = []
        for index, (g_obj, g_mask) in enumerate(zip(ground_truth.objects, truth_masks, strict=True)):
            if index in used_truth or a_obj.type != g_obj.type:
                continue
            intersection = int(np.logical_and(a_mask, g_mask).sum())
            if intersection:
                union = int(np.logical_or(a_mask, g_mask).sum())
                candidates.append((intersection / union, index, intersection, union))
        if candidates:
            iou, index, intersection, union = max(candidates)
            used_truth.add(index)
            pairs.append(ObjectScore(a_obj.nm, ground_truth.objects[index].nm, a_obj.type,
                                     intersection, union, iou))
        else:
            area = int(a_mask.sum())
            pairs.append(ObjectScore(a_obj.nm, None, a_obj.type, 0, area, 0.0))

    for index, (obj, mask) in enumerate(zip(ground_truth.objects, truth_masks, strict=True)):
        if index not in used_truth:
            area = int(mask.sum())
            pairs.append(ObjectScore(None, obj.nm, obj.type, 0, area, 0.0))

    intersection = sum(item.intersection for item in pairs)
    union = sum(item.union for item in pairs)
    return ComparisonResult(tuple(pairs), intersection, union)


def _to_mask(obj: AnnotationObject, width: int, height: int) -> np.ndarray:
    if obj.mask_path is not None:
        image = Image.open(obj.mask_path).convert("L")
        if image.size != (width, height):
            raise ValueError(f"mask {obj.mask_path} has incorrect dimensions")
        return np.asarray(image) > 0

    image = Image.new("1", (width, height))
    draw = ImageDraw.Draw(image)
    if obj.path:
        draw.polygon(obj.path, fill=1)
    else:
        x1, y1, x2, y2 = obj.bbox
        draw.rectangle((x1, y1, x2, y2), fill=1)
    return np.asarray(image, dtype=bool)
