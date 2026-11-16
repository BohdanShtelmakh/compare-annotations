from pathlib import Path

from src.annotation_comparator.comparison import compare
from src.annotation_comparator.parser import load_annotation


ROOT = Path(__file__).parent
annotated = load_annotation(ROOT / "data/annotated/road.jpg.json")
ground_truth = load_annotation(
    ROOT / "data/ground_truth/road.jpg.json",
    ROOT / "data/ground_truth/masks",
)
result = compare(annotated, ground_truth)

for item in result.objects:
    print(f"{item.annotated_nm or '-'} vs {item.ground_truth_nm or '-'}: "
          f"{item.object_type}, IoU={item.iou:.4f}")
print(f"Intersection: {result.intersection}")
print(f"Union: {result.union}")
print(f"Overall score: {result.score:.2f}%")
