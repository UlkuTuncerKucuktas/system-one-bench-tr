import sys

from .build import build
from .readout import readout
from .run import run
from .score import score


def main():
    command, *args = sys.argv[1:]
    {"build": build, "run": run, "readout": readout, "score": score}[command](*args)


if __name__ == "__main__":
    main()
