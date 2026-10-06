import argparse
from pathlib import Path

from . import STAGES, draw


def main():
    parser = argparse.ArgumentParser(description="Draw one figure group.")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    from reproduce import validate_output

    output = args.output.resolve()
    validate_output(output)
    draw(args.stage, output)


if __name__ == "__main__":
    main()
