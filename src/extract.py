from pathlib import Path

import cv2
import numpy as np


def extract_objects(
    image_dir: Path,
    label_dir: Path,
    output_dir: Path,
    padding: float = 0.25,
    iterations: int = 8,
):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError("Не найдены папки images или labels")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in image_extensions
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_objects = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Не удалось прочитать: {image_path.name}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Не найдена разметка: {label_path.name}")
            continue

        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        H, W = image.shape[:2]

        with open(label_path, "r") as file:
            for object_index, line in enumerate(file):
                if not line.strip():
                    continue

                class_id, x_center, y_center, width, height = map(float, line.split())

                class_id = int(class_id)

                x_center *= W
                y_center *= H
                width *= W
                height *= H

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_w = x2 - x1
                bbox_h = y2 - y1

                pad_x = max(10, int(bbox_w * padding))
                pad_y = max(10, int(bbox_h * padding))

                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(W, x2 + pad_x)
                crop_y2 = min(H, y2 + pad_y)

                crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

                h, w = crop.shape[:2]

                if h < 5 or w < 5:
                    continue

                rect_x = x1 - crop_x1
                rect_y = y1 - crop_y1

                rect_w = x2 - x1
                rect_h = y2 - y1

                if rect_w < 2 or rect_h < 2:
                    continue

                mask = np.zeros((h, w), np.uint8)

                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)

                try:
                    mask, bgd_model, fgd_model = cv2.grabCut(
                        crop,
                        mask,
                        (rect_x, rect_y, rect_w, rect_h),
                        bgd_model,
                        fgd_model,
                        iterations,
                        cv2.GC_INIT_WITH_RECT,
                    )

                except cv2.error as error:
                    print(f"Ошибка GrabCut для {image_path.name}: {error}")
                    continue

                mask_binary = np.where(
                    (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
                ).astype(np.uint8)

                alpha = mask_binary * 255

                object_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

                object_rgba[:, :, 3] = alpha

                object_path = image_output_dir / f"object_{object_index}.png"

                success = cv2.imwrite(str(object_path), object_rgba)

                if not success:
                    print(f"Не удалось сохранить: {object_path}")
                    continue

                extracted_objects.append((object_path, class_id))

        print(f"Обработано: {image_path.name}")

    return extracted_objects
