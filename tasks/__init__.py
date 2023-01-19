import tempfile
import pathlib
import invoke
from tqdm import tqdm
from . import cost, hpp, utils


@invoke.task
def lkh_build(ctx):
    """Download and build LKH-3."""
    version = "3.0.8"
    pathlib.Path("bin").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        ctx.run(f"wget -P {d} http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-{version}.tgz")
        ctx.run(f"tar -zxf {d}/LKH-{version}.tgz -C {d}")
        ctx.run(f"cd {d}/LKH-{version} && make")
        ctx.run(f"cp {d}/LKH-{version}/LKH ./bin/LKH")


@invoke.task
def hpp_orthant1(ctx, output_dir="outputs", initial_tour_file=None, runs=3, max_trials=100):
    """Solve Hamiltonian Path Problem for first orthant."""
    positions1, _, _, _ = hpp.split_positions()
    hpp.solve(
        name="orthant1",
        output_dir=output_dir,
        positions=positions1,
        source=(0, 64),
        sink=(0, 128),
        orthant=1,
        runs=runs,
        max_trials=max_trials,
        initial_tour_file=initial_tour_file,
    )


@invoke.task
def hpp_orthant2(ctx, output_dir="outputs", initial_tour_file=None, runs=3, max_trials=100):
    """Solve Hamiltonian Path Problem for second orthant."""
    _, positions2, _, _ = hpp.split_positions()
    hpp.solve(
        name="orthant2",
        output_dir=output_dir,
        positions=positions2,
        source=(0, 128),
        sink=(-128, 0),
        orthant=2,
        runs=runs,
        max_trials=max_trials,
        initial_tour_file=initial_tour_file,
    )


@invoke.task
def hpp_orthant3(ctx, output_dir="outputs", initial_tour_file=None, runs=3, max_trials=100):
    """Solve Hamiltonian Path Problem for third orthant."""
    _, _, positions3, _ = hpp.split_positions()
    hpp.solve(
        name="orthant3",
        output_dir=output_dir,
        positions=positions3,
        source=(-128, 0),
        sink=(0, -128),
        orthant=3,
        runs=runs,
        max_trials=max_trials,
        initial_tour_file=initial_tour_file,
    )


@invoke.task
def hpp_orthant4(ctx, output_dir="outputs", initial_tour_file=None, runs=3, max_trials=100):
    """Solve Hamiltonian Path Problem for fourth orthant."""
    _, _, _, positions4 = hpp.split_positions()
    hpp.solve(
        name="orthant4",
        output_dir=output_dir,
        positions=positions4,
        source=(128, -128),
        sink=(0, -64),
        orthant=4,
        runs=runs,
        max_trials=max_trials,
        initial_tour_file=initial_tour_file,
    )


@invoke.task
def merge_tours(ctx, orthant1, orthant2, orthant3, orthant4):
    """Merge the tours and generates submission file."""
    positions1, positions2, positions3, positions4 = hpp.split_positions()
    tour1 = hpp.load_tour(tour=orthant1, positions=positions1)
    tour2 = hpp.load_tour(tour=orthant2, positions=positions2)
    tour3 = hpp.load_tour(tour=orthant3, positions=positions3)
    tour4 = hpp.load_tour(tour=orthant4, positions=positions4)

    configurations1 = [hpp.position_to_config_tr(p) for p in tour1]
    configurations2 = [hpp.position_to_config_tl(p) for p in tour2]
    configurations2 = [configurations2[0]] + configurations2[1:][::-1]
    configurations3 = [hpp.position_to_config_bl(p) for p in tour3]
    configurations4 = [hpp.position_to_config_br(p) for p in tour4]

    c1, c2, c3 = hpp.gen_subconfig()

    configurations = c1[:-1] + configurations1[:-1] + configurations2[:-1] + configurations3[:-1] + c2[:-1] + configurations4[:-1] + c3
    utils.save_configurations(configurations=configurations, path="submissions/submission.csv")


@invoke.task
def evaluate(ctx, submission):
    """Calculates score.

    Parameters
    ----------
    submission:
        A path of submission file.
    """
    configurations = utils.load_configurations(submission)
    image = utils.load_image("data/image.csv")

    configuration_costs = []
    color_costs = []

    for i in tqdm(range(len(configurations)-1)):
        from_config = configurations[i]
        to_config = configurations[i+1]

        from_position = utils.config_to_position(from_config)
        to_position = utils.config_to_position(to_config)

        from_idx = utils.position_to_index(from_position)
        to_idx = utils.position_to_index(to_position)

        configuration_costs.append(cost.reconfiguration(from_config=from_config, to_config=to_config))
        color_costs.append(cost.color(from_idx=from_idx, to_idx=to_idx, image=image))

    configuration_cost = sum(configuration_costs)
    color_cost = sum(color_costs)
    print(f"Total Configuration Cost: {configuration_cost}")
    print(f"Total Color Cost: {color_cost}")
    print(f"Total Cost: {configuration_cost + color_cost}")
