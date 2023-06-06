import numpy as np

from colors import *

WHITE_COL = 255
BLACK_COL = 0


def identity(mask: np.ndarray) -> np.ndarray:
    return mask

def convert_to_binary(mask: np.ndarray, primary_color: ColorT) -> np.ndarray:
    res = np.zeros(mask.shape[0:2], dtype=np.uint8)
    res[np.where((mask == primary_color).all(axis=2))[0:2]] = WHITE_COL
    return res


def convert_to_binary_terrain(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=TERRAIN_COL)


def convert_to_binary_snow(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=SNOW_COL)


def convert_to_binary_sand(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=SAND_COL)


def convert_to_binary_forest(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=FOREST_COL)


def convert_to_binary_grass(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=GRASS_COL)


def convert_to_binary_roads(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=ROADS_COL)


def convert_to_binary_buildings(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=BUILDINGS_COL)


def convert_to_binary_water(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=WATER_COL)


def convert_to_binary_clouds(mask: np.ndarray) -> np.ndarray:
    return convert_to_binary(mask, primary_color=CLOUDS_COL)


def convert_from_binary(bin_mask: np.ndarray, primary_color: ColorT) -> np.ndarray:
    height, width = bin_mask.shape
    res = np.zeros((height, width, 3), dtype=np.uint8)
    res[np.where(bin_mask == WHITE_COL)] = primary_color
    return res


def convert_from_binary_terrain(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=TERRAIN_COL)


def convert_from_binary_snow(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=SNOW_COL)


def convert_from_binary_sand(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=SAND_COL)


def convert_from_binary_forest(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=FOREST_COL)


def convert_from_binary_grass(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=GRASS_COL)


def convert_from_binary_roads(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=ROADS_COL)


def convert_from_binary_buildings(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=BUILDINGS_COL)


def convert_from_binary_water(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=WATER_COL)


def convert_from_binary_clouds(mask: np.ndarray) -> np.ndarray:
    return convert_from_binary(mask, primary_color=CLOUDS_COL)


VALID_CLASSES = {'ground', 'grass', 'sand', 'snow', 'forest', 'roads', 'buildings', 'water', 'clouds'}

CLASS_TO_COL = {
    'ground': TERRAIN_COL,
    'grass': GRASS_COL,
    'sand': SAND_COL,
    'snow': SNOW_COL,
    'forest': FOREST_COL,
    'roads': ROADS_COL,
    'buildings': BUILDINGS_COL,
    'water': WATER_COL,
    'clouds': CLOUDS_COL
}

TO_BIN_CONVERTERS = {
    'ground': convert_to_binary_terrain,
    'grass': convert_to_binary_grass,
    'sand': convert_to_binary_sand,
    'snow': convert_to_binary_snow,
    'forest': convert_to_binary_forest,
    'roads': convert_to_binary_roads,
    'buildings': convert_to_binary_buildings,
    'water': convert_to_binary_water,
    'clouds': convert_to_binary_clouds
}

FROM_BIN_CONVERTERS = {
    'ground': convert_from_binary_terrain,
    'grass': convert_from_binary_grass,
    'sand': convert_from_binary_sand,
    'snow': convert_from_binary_snow,
    'forest': convert_from_binary_forest,
    'roads': convert_from_binary_roads,
    'buildings': convert_from_binary_buildings,
    'water': convert_from_binary_water,
    'clouds': convert_from_binary_clouds
}

# if __name__ == '__main__':
#     import cv2
#
#     bin_mask = cv2.imread('tmp/full_size/00_forest_mask.png', 0)
#     mask = convert_from_binary_forest(bin_mask)
#     cv2.imshow('aa', mask)
#     cv2.waitKey(0)
