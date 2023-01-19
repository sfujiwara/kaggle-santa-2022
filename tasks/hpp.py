import copy
import itertools
import math
import pathlib
import subprocess
from typing import Optional
import numpy as np
from tqdm import tqdm
from . import cost, utils


COST_SCALE = 1000


def split_positions() -> tuple:
    """Splits positions to four orthants."""
    positions1 = list(itertools.product(range(0, 129), range(1, 129)))
    positions1 = [p for p in positions1 if not (p[0] == 0 and p[1] < 64)]

    positions2 = [(0, 128)]
    positions2.extend(list(itertools.product(range(-128, 0), range(129))))

    positions3 = [(-128, 0)]
    positions3.extend(list(itertools.product(range(-128, 0), range(-128, 0))))
    positions3.append((0, -128))

    positions4 = [(128, -128)]
    positions4.extend(list(itertools.product(range(1, 129), range(-127, 1))))
    positions4.extend([(0, i) for i in range(-127, -63)])

    return positions1, positions2, positions3, positions4


def gen_subconfig():
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

    return path1, path2, path3


def position_to_config_tr(position):
    x, y = position[0], position[1]
    config = [[64, y-64], [-32, 32], [-16, 16], [-8, 8], [-4, 4], [-2, 2], [-1, 1], [-1, 1]]

    for i in range(x):
        if i % 2 == 0:
            config[1][0] += 1
        else:
            for j in range(2, 8):
                if config[j][0] < config[j][1]:
                    config[j][0] += 1
                    break
    else:
        return [tuple(arm) for arm in config]


def position_to_config_tl(position):
    """(_, 64), (-32, _), (-16, _) ,(-8, _), (-4, _), (-2, _), (-1, _), (-1, _)"""
    x, y = position[0], position[1]
    config = [[x+64, 64], [-32, -32], [-16, -16], [-8, -8], [-4, -4], [-2, -2], [-1, -1], [-1, -1]]
    for i in range(y):
        if i % 2 == 0:
            config[1][1] += 1
        else:
            for j in range(2, 8):
                if config[j][1] < -config[j][0]:
                    config[j][1] += 1
                    break
    else:
        return [tuple(arm) for arm in config]


def position_to_config_bl(position):
    """(-64, _), (_, -32), (_, -16) ,(_, -8), (_, -4), (_, -2), (_, -1), (_, -1)"""
    x, y = position[0], position[1]
    config = [[-64, 64+y], [32, -32], [16, -16], [8, -8], [4, -4], [2, -2], [1, -1], [1, -1]]
    for i in range(-x):
        if i % 2 == 0:
            config[1][0] -= 1
        else:
            for j in range(2, 8):
                if config[j][0] > config[j][1]:
                    config[j][0] -= 1
                    break
    else:
        return [tuple(arm) for arm in config]


def position_to_config_br(position):
    """(64, _), (_, -32), (_, -16) ,(_, -8), (_, -4), (_, -2), (_, -1), (_, -1)"""
    x, y = position[0], position[1]
    config = [[64, 64+y], [-32, -32], [-16, -16], [-8, -8], [-4, -4], [-2, -2], [-1, -1], [-1, -1]]

    for i in range(x):
        if i % 2 == 0:
            config[1][0] += 1
        else:
            for j in range(2, 8):
                if config[j][0] < -config[j][1]:
                    config[j][0] += 1
                    break
    else:
        return [tuple(arm) for arm in config]


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


def solve(
    name: str,
    output_dir: str,
    positions: list,
    source: tuple,
    sink: tuple,
    orthant: int,
    runs: int,
    max_trials: int,
    initial_tour_file: Optional[str] = None,
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    image = utils.load_image()

    indices = [utils.position_to_index(p) for p in positions]
    length = len(positions)

    adjmat = np.zeros([length, length], dtype=int)

    if orthant == 1 or orthant == 3 or orthant == 4:
        x_ = 1
        y_ = 0
    else:
        x_ = 0
        y_ = 1

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
            elif step_x >= 2 + x_ or step_y >= 2 + y_:
                adjmat[i, j] = adjmat[j, i] = 30000
            else:
                color_cost = cost.color(from_idx=idx1, to_idx=idx2, image=image)
                conf_cost = math.sqrt(step_x + step_y)
                adjmat[i, j] = adjmat[j, i] = int((color_cost + conf_cost) * COST_SCALE)

    print("Writing TSP file...")
    with open(f"{output_dir}/{name}.tsp", "w") as f:
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
    with open(f"{output_dir}/{name}.par", "w") as f:
        f.write(f"PROBLEM_FILE = {output_dir}/{name}.tsp\n")
        f.write(f"OUTPUT_TOUR_FILE = {output_dir}/{name}-$.tour\n")
        f.write(f"TOUR_FILE = {output_dir}/{name}.tour\n")
        # f.write("TIME_LIMIT = 60000\n")
        f.write("TRACE_LEVEL = 1\n")
        f.write(f"MAX_TRIALS = {max_trials}\n")
        f.write("CANDIDATE_SET_TYPE = POPMUSIC\n")
        # f.write("POPMUSIC_SAMPLE_SIZE = 100\n")
        # f.write("POPMUSIC_TRIALS = 0\n")
        # f.write("MAX_CANDIDATES = 100\n")
        f.write(f"RUNS = {runs}\n")
        f.write("SEED = 42\n")
        if initial_tour_file:
            f.write(f"INITIAL_TOUR_FILE = {initial_tour_file}\n")
        f.write("EOF\n")

    subprocess.run([f"./bin/LKH", f"{output_dir}/{name}.par"])


def load_tour(tour, positions):
    with open(tour) as f:
        tour = [int(i) - 1 for i in f.readlines()[6:-2]]

    tour_p = [positions[t] for t in tour]

    return tour_p
