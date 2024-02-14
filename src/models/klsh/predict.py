import argparse
import json
import pathlib
import random
import subprocess

from unique_names_generator import get_random_name


def get_git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


parser = argparse.ArgumentParser()

parser.add_argument(
    "--database",
    "-d",
    default="RLDATA500-coffee_mole",
    type=str,
)

parser.add_argument(
    "--number_of_random_projections",
    default=100,
    type=int,
)

parser.add_argument(
    "--number_of_blocks",
    default=5,
    type=int,
    help="desired number of blocks of the output",
)

parser.add_argument(
    "--shingle_size",
    default=2,
    type=int,
)

parser.add_argument(
    "--seed",
    default=0,
    type=int,
)

args = parser.parse_args()

name_seeed = dict(vars(args))
name_seeed["database"] = None  # database shouldn't affect name
name_seeed = json.dumps(name_seeed)
random.seed(name_seeed)
unique_name = get_random_name(separator="_", style="lowercase")

OUTPUT_DIR = pathlib.Path(f"research/models/klsh/{args.database}/{unique_name}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

subproces_args = [
    "src/models/klsh/r/klsh.R",
    args.database,
    args.number_of_blocks,
    args.number_of_random_projections,
    args.shingle_size,
    args.seed,
    OUTPUT_DIR,
]
print(subproces_args)
subprocess.run([str(x) for x in subproces_args])


with (OUTPUT_DIR / "hparams.json").open("w") as f:
    hparams = vars(args)
    hparams["model"] = "klsh"
    hparams["commit"] = get_git_revision()
    f.write(json.dumps(hparams, indent=True))
