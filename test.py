"""FWI evaluation entry point; run `uv run test.py --help`."""
import sys

from tap_syn.cli import main


if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
