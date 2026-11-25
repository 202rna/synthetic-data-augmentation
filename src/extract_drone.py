from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path("data/original/")
CONCRETE_DIR = "proba"
STORE_DIR = Path("data/processed")

# Пути к исходным данным
IMAGE_DIR = DATA_DIR / CONCRETE_DIR / "images"
LABEL_DIR = DATA_DIR / CONCRETE_DIR / "labels"

OUTPUT_DIR = STORE_DIR / CONCRETE_DIR

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
    raise FileNotFoundError("Не найдены папки images или labels")


image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in image_extensions
)

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    with open(label_path, "r") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())

            class_id = int(class_id)

            print(class_id, x_center, y_center, width, height)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            H, W = image.shape[:2]

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

            image_box = image_rgb.copy()

            cv2.rectangle(image_box, (x1, y1), (x2, y2), (255, 0, 0), 2)

            crop = image[y1:y2, x1:x2]

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            mask = np.zeros(crop.shape[:2], np.uint8)
            h, w = crop.shape[:2]

            rect = (
                2,
                2,
                w - 4,
                h - 4
            )
            
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            mask, bgd_model, fgd_model = cv2.grabCut(
                crop,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT
            )
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            mask, bgd_model, fgd_model = cv2.grabCut(
                crop,
                mask,
                rect,
                bgd_model,
                fgd_model,
                5,
                cv2.GC_INIT_WITH_RECT
            )
            mask_binary = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                1,
                0
            ).astype(np.uint8)
            result = crop * mask_binary[:, :, np.newaxis]

            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            alpha = (mask_binary * 255).astype(np.uint8)

            drone_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)

            drone_rgba[:, :, 3] = alpha
            image_output_dir = OUTPUT_DIR / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)

            # cv2.imwrite("drone_cutout.png", drone_rgba)
            # saved = cv2.imread("drone_cutout.png", cv2.IMREAD_UNCHANGED)
                
            image_output_dir = Path(image_output_dir)
            image_output_dir.mkdir(parents=True, exist_ok=True)

            path_to_save = image_output_dir / f"{x_center}_{y_center}.png"

            cv2.imwrite(
                str(path_to_save),
                drone_rgba
            )
            
            saved = cv2.imread(
                path_to_save,
                cv2.IMREAD_UNCHANGED,
            )

            plt.imshow(cv2.cvtColor(saved, cv2.COLOR_BGRA2RGBA))
            plt.axis("off")
            plt.show()
