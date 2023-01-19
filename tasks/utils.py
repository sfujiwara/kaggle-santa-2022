import numpy as np
import pandas as pd


def config_to_str(config):
    s = str(config)
    s = s.replace("), ", ";").replace(", ", " ").replace("(", "").replace("[", "").replace(")", "").replace("]", "")
    return s


def load_configurations(path: str) -> list:
    df = pd.read_csv(path)
    configurations = []

    for row in df.itertuples():
        tmp = row.configuration.split(";")
        tmp = [i.split(" ") for i in tmp]
        configuration = [(int(i[0]), int(i[1])) for i in tmp]
        configurations.append(configuration)

    return configurations


def save_configurations(path: str, configurations):
    with open(path, "w") as f:
        f.write("configuration\n")
        for c in configurations:
            f.write(f"{config_to_str(c)}\n")


def load_image(path: str = "data/image.csv"):
    df = pd.read_csv(path)
    side = int(len(df) ** 0.5)  # assumes a square image
    return df.set_index(["x", "y"]).to_numpy().reshape(side, side, -1)


def load_tour(path: str) -> list[tuple[int, int]]:
    with open(path) as f:
        tour = [int(i) for i in f.readlines()[6:-2]]

    start = tour.index(33025)
    tour = tour[start:] + tour[:start]
    indices = [((t-1) // 257, (t-1) % 257) for t in tour]
    positions = [index_to_position(i) for i in indices]

    return positions


def config_to_position(config) -> tuple[int, int]:
    if len(config) == 0:
        return 0, 0

    position = np.array(config).sum(axis=0)

    return tuple(position.tolist())


def position_to_index(position, shape=(257, 257)):
    m, n = shape[:2]
    x, y = position[0], position[1]
    i = (n - 1) // 2 - y
    j = (n - 1) // 2 + x
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    return i, j


def index_to_position(index, shape=(257, 257)):
    m, n = shape[:2]
    i, j = index[0], index[1]
    if i < 0 or i >= m or j < 0 or j >= n:
        raise ValueError("Coordinates not within given dimensions.")
    y = (n - 1) // 2 - i
    x = j - (n - 1) // 2
    return x, y


def node_id_to_index(node_id: int, size: int = 257) -> tuple[int, int]:
    return (node_id-1) // size, (node_id-1) % size


def node_id_to_position(node_id: int, size: int = 257) -> tuple[int, int]:
    index = node_id_to_index(node_id)
    position = index_to_position(index=index)
    return position
