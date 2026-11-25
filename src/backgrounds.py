import random
from enum import Enum

import numpy as np


class Colors(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    RED = (0, 0, 255)
    GREEN = (0, 128, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PURPLE = (128, 0, 128)
    CYAN = (255, 255, 0)


def generate_backgrounds(count, width=1280, height=720, available_colors=None):
    if count < 1:
        raise ValueError("Количество фонов должно быть больше нуля")

    if available_colors is None:
        available_colors = list(Colors)

    if not available_colors:
        raise ValueError("Список цветов не может быть пустым")

    backgrounds = []

    for _ in range(count):
        color = random.choice(available_colors)

        background = np.full((height, width, 3), color.value, dtype=np.uint8)

        backgrounds.append(background)

    return backgrounds
