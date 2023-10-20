'''

'''

import taichi as ti

class SPECIES:
    SIZE_MIN        = 2.5
    SIZE_SCALE      = 2.5
    SPEED_MIN       = 0.1
    SPEED_SCALE     = 1.0
    MAX_SPEED_MIN   = 1.0
    MAX_SPEED_SCALE = 3.0
    MASS_MIN        = 1.0
    MASS_SCALE      = 5.0
    DECAY_MIN       = 0.9
    DECAY_SCALE     = 0.099

class VERA:
    TEST = 0.1

class BOIDS:
    SEP_MIN    = 0.1
    COH_MIN    = 0.1
    ALI_MIN    = 0.1
    RADIUS_MAX = 300.0

class PHYSARUM:
    SENSE_ANGLE = 0.3 * ti.math.pi
    MOVE_ANGLE  = 0.3 * ti.math.pi
    SENSE_DIST  = 50.0
    MOVE_DIST   = 1.0
    DIST_MIN    = 0.1
    # EVAP_MIN  = 0.5

class ATTRACTOR:
    POS_MIN    = 0.2
    POS_MAX    = 0.8
    MASS_SCALE = 1.0
    SPEED      = 1.0
    MAX_SPEED  = 2.0

