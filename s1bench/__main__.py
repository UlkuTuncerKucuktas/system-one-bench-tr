import sys

from .build import build
from .run import run
from .score import score


def main():
    command, *args = sys.argv[1:]
    {"build": build, "run": run, "score": score}[command](*args)


if __name__ == "__main__":
    main()
