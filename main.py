from pathlib import Path

import cv2

from src.backgrounds import generate_backgrounds
from src.compose import compose_object
from src.extract import extract_objects

# =====================================
# CONFIGURATION
# =====================================

DATA_DIR = Path("data/original")
STORE_DIR = Path("data/processed")

CONCRETE_DIR = "proba"

IMAGE_DIR = DATA_DIR / CONCRETE_DIR / "images"
LABEL_DIR = DATA_DIR / CONCRETE_DIR / "labels"

OBJECTS_DIR = STORE_DIR / CONCRETE_DIR / "objects"

SYNTHETIC_DIR = STORE_DIR / CONCRETE_DIR / "synthetic"
SYNTHETIC_IMAGE_DIR = SYNTHETIC_DIR / "images"
SYNTHETIC_LABEL_DIR = SYNTHETIC_DIR / "labels"

BACKGROUND_COUNT = 3

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# =====================================
# MAIN PIPELINE
# =====================================


def main():

    print("\n=== Synthetic Data Augmentation ===\n")

    print("[1/3] Extracting objects...")

    extracted_objects = extract_objects(
        image_dir=IMAGE_DIR, label_dir=LABEL_DIR, output_dir=OBJECTS_DIR
    )

    print(f"Extract objects: {len(extracted_objects)}")

    if not extracted_objects:
        print("Objeacts don't found. End of generation.")
        return

    print("\n[2/3] Background generation...")

    backgrounds = generate_backgrounds(
        count=BACKGROUND_COUNT, width=IMAGE_WIDTH, height=IMAGE_HEIGHT
    )

    SYNTHETIC_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[3/3] Creating syntetic images...")

    total = 0

    for object_index, (object_path, class_id) in enumerate(extracted_objects, start=1):
        object_rgba = cv2.imread(str(object_path), cv2.IMREAD_UNCHANGED)

        if object_rgba is None:
            print(f"Can't read the path: {object_path}")
            continue

        for background_index, background in enumerate(backgrounds, start=1):
            output_name = (
                f"{object_path.parent.name}_"
                f"{object_path.stem}_"
                f"{background_index:03d}.png"
            )

            image_path = SYNTHETIC_IMAGE_DIR / output_name
            label_path = SYNTHETIC_LABEL_DIR / f"{image_path.stem}.txt"

            bbox = compose_object(
                background=background, object_rgba=object_rgba, output_path=image_path
            )

            x_center, y_center, width, height = bbox

            with open(label_path, "w") as file:
                file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}\n"
                )

            total += 1

        print(f"[{object_index}/{len(extracted_objects)}] {object_path.name}")

    print("\n=== Result ===")
    print(f"Extract objeacts: {len(extracted_objects)}")
    print(f"Background counts: {BACKGROUND_COUNT}")
    print(f"Created images: {total}")
    print(f"Images path: {SYNTHETIC_IMAGE_DIR.resolve()}")
    print(f"Labels path: {SYNTHETIC_LABEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
