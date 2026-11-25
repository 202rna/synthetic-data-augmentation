from pathlib import Path

import cv2
import numpy as np


def compose_object(background, object_rgba, output_path: Path):
    """
    Накладывает объект с альфа-каналом на фон,
    сохраняет изображение и возвращает YOLO-координаты bbox.
    """

    if object_rgba is None or object_rgba.ndim != 3 or object_rgba.shape[2] != 4:
        raise ValueError("Объект должен быть PNG с альфа-каналом")

    result = background.copy()

    bg_h, bg_w = result.shape[:2]
    obj_h, obj_w = object_rgba.shape[:2]

    if obj_h > bg_h or obj_w > bg_w:
        raise ValueError("Объект больше фона. Нужно уменьшить объект.")

    x = (bg_w - obj_w) // 2
    y = (bg_h - obj_h) // 2

    roi = result[y : y + obj_h, x : x + obj_w]

    object_bgr = object_rgba[:, :, :3]
    alpha = object_rgba[:, :, 3:4] / 255.0

    roi[:] = (object_bgr * alpha + roi * (1 - alpha)).astype(np.uint8)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), result)

    if not success:
        raise RuntimeError(f"Не удалось сохранить изображение: {output_path}")

    x_center = (x + obj_w / 2) / bg_w
    y_center = (y + obj_h / 2) / bg_h

    width = obj_w / bg_w
    height = obj_h / bg_h

    return x_center, y_center, width, height
