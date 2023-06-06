from typing import Tuple, Dict

ColorT = Tuple[int, int, int]

TERRAIN_COL = (0., 51., 102.)
SNOW_COL = (255., 255., 204.)
SAND_COL = (51., 255., 255.)
FOREST_COL = (0., 102., 0.)
GRASS_COL = (51., 255., 51.)

ROADS_COL = (160., 160., 160.)

BUILDINGS_COL = (96., 96., 96.)

WATER_COL = (255., 128., 0.)

CLOUDS_COL = (224., 224., 224.)

UNKNOWN_COL = (0., 0., 0.)

COLORS = {TERRAIN_COL,
          SNOW_COL,
          SAND_COL,
          FOREST_COL,
          GRASS_COL,
          ROADS_COL,
          BUILDINGS_COL,
          WATER_COL,
          CLOUDS_COL}

COLOR_2_TYPE = \
    {TERRAIN_COL: 0,
     SNOW_COL: 1,
     SAND_COL: 2,
     FOREST_COL: 3,
     GRASS_COL: 4,
     ROADS_COL: 5,
     BUILDINGS_COL: 6,
     WATER_COL: 7,
     CLOUDS_COL: 8,
     UNKNOWN_COL: 9}

TYPE_2_COLOR = \
    {0: TERRAIN_COL,
     1: SNOW_COL,
     2: SAND_COL,
     3: FOREST_COL,
     4: GRASS_COL,
     5: ROADS_COL,
     6: BUILDINGS_COL,
     7: WATER_COL,
     8: CLOUDS_COL,
     9: UNKNOWN_COL}

COL_TO_CLASS = {
    TERRAIN_COL: 'ground',
    GRASS_COL: 'grass',
    SAND_COL: 'sand',
    SNOW_COL: 'snow',
    FOREST_COL: 'forest',
    ROADS_COL: 'roads',
    BUILDINGS_COL: 'buildings',
    WATER_COL: 'water',
    CLOUDS_COL: 'clouds'
}