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
    if item.annotated_nm and item.ground_truth_nm:
        status = "MATCHED"
        names = f"{item.annotated_nm} vs {item.ground_truth_nm}"
    elif item.annotated_nm:
        status = "MISSED OBJECT"
        names = f"{item.annotated_nm} vs -"
    else:
        status = "EXTRA ANNOTATION"
        names = f"- vs {item.ground_truth_nm}"
    print(f"{status}: {names}, {item.object_type}, "
          f"IoU={item.iou:.4f}, intersection={item.intersection}, union={item.union}")
print(f"Intersection: {result.intersection}")
print(f"Union: {result.union}")
print(f"Overall score: {result.score:.2f}%")
