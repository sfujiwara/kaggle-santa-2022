import copy
import itertools
import numpy as np
from tqdm import tqdm
import math
from . import utils
from . import cost
import subprocess


COST_SCALE = 100


def standard_config_topright(position):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    x, y = position[0], position[1]
    # assert x > 0 and y >= 0, "This function is only for the upper right quadrant"
    r = 64
    config = [(r, y-r)]  # longest arm points to the right
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, r))  # arm points upwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, r))  # arm points upwards
    assert x == arm_x
    return config


def standard_config_topleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    # assert x <= 0 and y > 0, "This function is only for the upper left quadrant"
    # (_, 64), (-32, _), (-16, _) ,(-8, _), (-4, _), (-2, _), (-1, _), (-1, _)
    r = 64
    config = [(x-(-r), r)] # longest arm points to the top
    y = y - config[0][1]
    while r > 1:
        r = r // 2
        arm_y = np.clip(y, -r, r)
        config.append((-r, arm_y))  # arm points leftwards
        y -= arm_y
    arm_y = np.clip(y, -r, r)
    config.append((-r, arm_y))  # arm points leftwards
    assert y == arm_y
    return config


def standard_config_bottomright(position):
    # [(64, _), (_, -32), (_, -16), (_, -8), (_, -4), (_, -2), (_, -1), (_, -1)]
    x, y = position[0], position[1]
    assert y <= 0 <= x

    r = 64
    config = [(r, y+r)]  # longest arm points to the right
    x = x - config[0][0]

    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, -r))  # arm points upwards
        x -= arm_x

    arm_x = np.clip(x, -r, r)
    config.append((arm_x, -r))  # arm points upwards

    assert x == arm_x

    return config


def standard_config_bottomleft(x, y):
    """Return the preferred configuration (list of eight pairs) for the point (x,y)"""
    # assert x < 0 and y <= 0, "This function is only for the lower left quadrant"
    # (-64, _),(_, -32), (_, -16) ,(_, -8), (_, -4), (_, -2), (_, -1), (_, -1)
    r = 64
    config = [(-r, y-(-r))]  # longest arm points to the left
    x = x - config[0][0]
    while r > 1:
        r = r // 2
        arm_x = np.clip(x, -r, r)
        config.append((arm_x, -r))  # arm points downwards
        x -= arm_x
    arm_x = np.clip(x, -r, r)
    config.append((arm_x, -r))  # arm points downwards
    assert x == arm_x
    return config


def solve_hpp(name: str, positions: list, source: tuple, sink: tuple):
    image = utils.load_image()

    indices = [utils.position_to_index(p) for p in positions]
    length = len(positions)

    adjmat = np.zeros([length, length], dtype=int)

    print("Computing edge costs...")
    for i in tqdm(range(length)):
        for j in range(i, length):
            pos1, pos2 = positions[i], positions[j]
            idx1, idx2 = indices[i], indices[j]
            step_x = abs(pos2[0]-pos1[0])
            step_y = abs(pos2[1]-pos1[1])
            if (pos1 == source and pos2 == sink) or (pos1 == sink and pos2 == source):
                print(pos1, pos2)
                adjmat[i, j] = adjmat[j, i] = -50000
                # import IPython;IPython.embed()
            elif step_x >= 2 or step_y >= 2:
                adjmat[i, j] = adjmat[j, i] = 30000
            else:
                color_cost = cost.color(from_idx=idx1, to_idx=idx2, image=image)
                conf_cost = math.sqrt(step_x + step_y)
                adjmat[i, j] = adjmat[j, i] = int((color_cost + conf_cost) * COST_SCALE)

    print("Writing TSP file...")
    with open(f"data/{name}.tsp", "w") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {length}\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW\n")
        f.write("EDGE_DATA_FORMAT: EDGE_LIST\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i, row in tqdm(enumerate(adjmat)):
            f.write(f"{' '.join(map(str, row.tolist()[i:]))}\n")

    print("Writing parameter file...")
    with open(f"data/{name}.par", "w") as f:
        f.write(f"PROBLEM_FILE = data/{name}.tsp\n")
        f.write(f"OUTPUT_TOUR_FILE = data/{name}-$.tour\n")
        f.write(f"TOUR_FILE = data/{name}.tour\n")
        # f.write("TIME_LIMIT = 60000\n")
        f.write("TRACE_LEVEL = 1\n")
        f.write("MAX_TRIALS = 30\n")
        f.write("CANDIDATE_SET_TYPE = POPMUSIC\n")
        f.write("POPMUSIC_SAMPLE_SIZE = 100\n")
        f.write("POPMUSIC_TRIALS = 0\n")
        f.write("MAX_CANDIDATES = 100\n")
        f.write("RUNS = 3\n")
        f.write("SEED = 42\n")
        # f.write("INITIAL_TOUR_FILE = data/orthant1-1762296.tour\n")
        f.write("EOF\n")

    subprocess.run([f"./bin/LKH", f"data/{name}.par"])

    with open(f"data/{name}.tour") as f:
        tour = [int(i) - 1 for i in f.readlines()[6:-2]]

    tour_p = [positions[t] for t in tour]

    return tour_p


def load_tour(name, positions):
    with open(f"data/{name}.tour") as f:
        tour = [int(i) - 1 for i in f.readlines()[6:-2]]

    tour_p = [positions[t] for t in tour]

    return tour_p


# Orthant 1.
positions1 = list(itertools.product(range(0, 129), range(1, 129)))
positions1 = [p for p in positions1 if not (p[0] == 0 and p[1] < 64)]
tour1 = solve_hpp(name="orthant1", positions=positions1, source=(0, 64), sink=(0, 128))
# tour1 = load_tour(name="orthant1-1762296", positions=positions1)

# Orthant 2.
positions2 = [(0, 128)]
positions2.extend(list(itertools.product(range(-128, 0), range(129))))
# tour2 = solve_hpp(name="orthant2", positions=positions2, source=(0, 128), sink=(-128, 0))
tour2 = load_tour(name="orthant2-1807109", positions=positions2)

# Orthant 3.
positions3 = [(-128, 0)]
positions3.extend(list(itertools.product(range(-128, 0), range(-128, 0))))
positions3.append((0, -128))
# tour3 = solve_hpp(name="orthant3", positions=positions3, source=(-128, 0), sink=(0, -128))
tour3 = load_tour(name="orthant3-1798212", positions=positions3)

# Orthant 4.
positions4 = [(128, -128)]
positions4.extend(list(itertools.product(range(1, 129), range(-127, 1))))
positions4.extend([(0, i) for i in range(-127, -63)])
# tour4 = solve_hpp(name="orthant4", positions=positions4, source=(128, -128), sink=(0, -64))
tour4 = load_tour(name="orthant4-1818784", positions=positions4)

configurations1 = [standard_config_topright(p) for p in tour1]
configurations2 = [standard_config_topleft(p[0], p[1]) for p in tour2]
configurations2 = [configurations2[0]] + configurations2[1:][::-1]
configurations3 = [standard_config_bottomleft(p[0], p[1]) for p in tour3]
configurations4 = [standard_config_bottomright(p) for p in tour4]

# Path from (0, 0) to (0, 64).
path1 = [[(64, 0), (-32, 0), (-16, 0), (-8, 0), (-4, 0), (-2, 0), (-1, 0), (-1, 0)]]
for _ in range(64):
    c = copy.deepcopy(path1[-1])
    for i in range(1, 8):
        if c[i][1] < -c[i][0]:
            c[i] = (c[i][0], c[i][1] + 1)
            break
    path1.append(c)

# Path from (0, -128) to (128, -128).
path2 = [
    [(i, -64), (32, -32), (16, -16), (8, -8), (4, -4), (2, -2), (1, -1), (1, -1)]
    for i in range(-64, 65)
]

# Path from (0, -64) to (0, 0).
path3 = [[(64, 0), (-32, -32), (-16, -16), (-8, -8), (-4, -4), (-2, -2), (-1, -1), (-1, -1)]]
for _ in range(64):
    c = copy.deepcopy(path3[-1])
    for i in range(1, 8):
        if c[i][1] < 0:
            c[i] = (c[i][0], c[i][1] + 1)
            break
    path3.append(c)

configurations = path1[:-1] + configurations1[:-1] + configurations2[:-1] + configurations3[:-1] + path2[:-1] + configurations4[:-1] + path3

utils.save_configurations(configurations=configurations, path="submissions/submission.csv")

import IPython;IPython.embed()

# 1762346
# 1807128
# 1812764
# 1825175
