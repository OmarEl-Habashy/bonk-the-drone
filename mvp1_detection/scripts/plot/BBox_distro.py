from pathlib import Path
from mcos_decoder import load_groundtruth
from collections import Counter, defaultdict
import statistics
import csv

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT

VIDEO_V_ROOT = Path("Drone-detection-2021-dataset\Data\Video_V")

OUTPUT_CSV = "drone_detection_2021_bbox_distribution.csv"


def classify_bbox(relative_area):
    if relative_area < 0.001:
        return "tiny"
    elif relative_area < 0.01:
        return "small"
    elif relative_area < 0.05:
        return "medium"
    else:
        return "large"


def infer_class_from_filename(path: Path):
    name = path.name.upper()

    if "DRONE" in name:
        return "drone"
    if "AIRPLANE" in name:
        return "airplane"
    if "BIRD" in name:
        return "bird"
    if "HELICOPTER" in name:
        return "helicopter"

    return "unknown"


def summarize(values):
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    mat_files = sorted(VIDEO_V_ROOT.rglob("*_LABELS.mat"))

    print(f"Found {len(mat_files)} label files")

    global_stats = {
        "widths": [],
        "heights": [],
        "relative_areas": [],
        "buckets": [],
        "valid_boxes": 0,
        "absent_frames": 0,
        "total_frames": 0,
    }

    class_stats = defaultdict(lambda: {
        "files": 0,
        "widths": [],
        "heights": [],
        "relative_areas": [],
        "buckets": [],
        "valid_boxes": 0,
        "absent_frames": 0,
        "total_frames": 0,
    })

    per_file_rows = []

    for mat_path in mat_files:
        object_class = infer_class_from_filename(mat_path)

        try:
            bboxes = load_groundtruth(str(mat_path))
        except Exception as e:
            print(f"[ERROR] Failed to parse {mat_path}: {e}")
            continue

        file_widths = []
        file_heights = []
        file_relative_areas = []
        file_buckets = []
        file_absent = 0

        for bbox in bboxes:
            global_stats["total_frames"] += 1
            class_stats[object_class]["total_frames"] += 1

            if bbox is None:
                file_absent += 1
                global_stats["absent_frames"] += 1
                class_stats[object_class]["absent_frames"] += 1
                continue

            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                file_absent += 1
                global_stats["absent_frames"] += 1
                class_stats[object_class]["absent_frames"] += 1
                continue

            relative_area = (w * h) / IMAGE_AREA
            bucket = classify_bbox(relative_area)

            file_widths.append(w)
            file_heights.append(h)
            file_relative_areas.append(relative_area)
            file_buckets.append(bucket)

            global_stats["widths"].append(w)
            global_stats["heights"].append(h)
            global_stats["relative_areas"].append(relative_area)
            global_stats["buckets"].append(bucket)
            global_stats["valid_boxes"] += 1

            class_stats[object_class]["widths"].append(w)
            class_stats[object_class]["heights"].append(h)
            class_stats[object_class]["relative_areas"].append(relative_area)
            class_stats[object_class]["buckets"].append(bucket)
            class_stats[object_class]["valid_boxes"] += 1

        class_stats[object_class]["files"] += 1

        area_summary = summarize(file_relative_areas)
        bucket_counts = Counter(file_buckets)

        per_file_rows.append({
            "file": str(mat_path),
            "class": object_class,
            "total_frames": len(bboxes),
            "valid_boxes": len(file_relative_areas),
            "absent_frames": file_absent,
            "mean_area_percent": area_summary["mean"] * 100 if area_summary["mean"] is not None else None,
            "median_area_percent": area_summary["median"] * 100 if area_summary["median"] is not None else None,
            "min_area_percent": area_summary["min"] * 100 if area_summary["min"] is not None else None,
            "max_area_percent": area_summary["max"] * 100 if area_summary["max"] is not None else None,
            "tiny_count": bucket_counts["tiny"],
            "small_count": bucket_counts["small"],
            "medium_count": bucket_counts["medium"],
            "large_count": bucket_counts["large"],
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file",
            "class",
            "total_frames",
            "valid_boxes",
            "absent_frames",
            "mean_area_percent",
            "median_area_percent",
            "min_area_percent",
            "max_area_percent",
            "tiny_count",
            "small_count",
            "medium_count",
            "large_count",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_file_rows)

    print("\n" + "=" * 70)
    print("GLOBAL RGB / VISIBLE BBOX DISTRIBUTION")
    print("=" * 70)

    print_summary("all_classes", global_stats)

    print("\n" + "=" * 70)
    print("PER-CLASS RGB / VISIBLE BBOX DISTRIBUTION")
    print("=" * 70)

    for object_class, stats in class_stats.items():
        print_summary(object_class, stats)

    print(f"\nSaved per-file CSV to: {OUTPUT_CSV}")


def print_summary(name, stats):
    print(f"\nClass: {name}")
    print("-" * 50)

    print(f"Files: {stats.get('files', 'N/A')}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Valid bbox frames: {stats['valid_boxes']}")
    print(f"Absent / invalid frames: {stats['absent_frames']}")

    if not stats["relative_areas"]:
        print("No valid boxes.")
        return

    width_summary = summarize(stats["widths"])
    height_summary = summarize(stats["heights"])
    area_summary = summarize(stats["relative_areas"])

    print("\nWidth:")
    print(f"Mean: {width_summary['mean']:.2f}px")
    print(f"Median: {width_summary['median']:.2f}px")
    print(f"Min: {width_summary['min']:.2f}px")
    print(f"Max: {width_summary['max']:.2f}px")

    print("\nHeight:")
    print(f"Mean: {height_summary['mean']:.2f}px")
    print(f"Median: {height_summary['median']:.2f}px")
    print(f"Min: {height_summary['min']:.2f}px")
    print(f"Max: {height_summary['max']:.2f}px")

    print("\nRelative area:")
    print(f"Mean: {area_summary['mean']:.6f}")
    print(f"Median: {area_summary['median']:.6f}")
    print(f"Min: {area_summary['min']:.6f}")
    print(f"Max: {area_summary['max']:.6f}")

    print("\nRelative area as percentage:")
    print(f"Mean: {area_summary['mean'] * 100:.4f}%")
    print(f"Median: {area_summary['median'] * 100:.4f}%")
    print(f"Min: {area_summary['min'] * 100:.4f}%")
    print(f"Max: {area_summary['max'] * 100:.4f}%")

    bucket_counts = Counter(stats["buckets"])
    total = len(stats["buckets"])

    print("\nSize buckets:")
    for bucket in ["tiny", "small", "medium", "large"]:
        count = bucket_counts[bucket]
        percentage = count / total * 100
        print(f"{bucket}: {count} boxes ({percentage:.2f}%)")


if __name__ == "__main__":
    main()