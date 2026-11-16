import argparse
import json
from pathlib import Path

from src.annotation_comparator.comparison import compare
from src.annotation_comparator.parser import load_annotation

ROOT = Path(__file__).parent


def run(annotated_path: Path, ground_truth_path: Path, masks_dir: Path, output_path: Path) -> None:
    annotated = load_annotation(annotated_path)
    ground_truth = load_annotation(ground_truth_path, masks_dir)
    result = compare(annotated, ground_truth)

    objects = []
    for item in result.objects:
        if item.annotated_nm and item.ground_truth_nm:
            status = "MATCHED"
            names = f"{item.annotated_nm} vs {item.ground_truth_nm}"
        elif item.annotated_nm:
            status = "EXTRA ANNOTATION"
            names = f"{item.annotated_nm} vs -"
        else:
            status = "MISSED OBJECT"
            names = f"- vs {item.ground_truth_nm}"
        print(f"{status}: {names}, {item.object_type}, "
              f"IoU={item.iou:.4f}, intersection={item.intersection}, union={item.union}")
        objects.append({
            "status": status,
            "annotated_nm": item.annotated_nm,
            "ground_truth_nm": item.ground_truth_nm,
            "type": item.object_type,
            "iou": item.iou,
            "intersection": item.intersection,
            "union": item.union,
        })
    print(f"Intersection: {result.intersection}")
    print(f"Union: {result.union}")
    print(f"Overall score: {result.score:.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "score": result.score,
        "intersection": result.intersection,
        "union": result.union,
        "objects": objects,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image annotations with ground truth.")
    parser.add_argument("--annotated", type=Path, default=ROOT / "data/annotated/road.jpg.json")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "data/ground_truth/road.jpg.json")
    parser.add_argument("--masks-dir", type=Path, default=ROOT / "data/ground_truth/masks")
    parser.add_argument("--output", type=Path, default=ROOT / "output/results.json")
    args = parser.parse_args()
    run(args.annotated, args.ground_truth, args.masks_dir, args.output)


if __name__ == "__main__":
    main()
