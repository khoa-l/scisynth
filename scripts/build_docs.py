"""Build the API docs with pdoc into ``docs/api``.

Run it with ``uv run python scripts/build_docs.py``. The default is the public
API: one page for each subpackage, listing the names it exports (its
``__all__``), in the order of that list.

With ``--internals`` it documents every module instead, nested by folder. pdoc
follows ``__all__``, which hides the modules inside a package, so the lists are
dropped for that build, in this process only. The package is not changed.
"""

import argparse
import importlib
from pathlib import Path

import pdoc
import pdoc.render

PACKAGES = [
    "core",
    "latent",
    "samplers",
    "stages",
    "observer",
    "projection",
    "viz",
    "testing",
]
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "api"
PRIVATE = r"!.*\._"  # a regular expression: leave out the underscore modules


def main() -> None:
    """Write the docs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--internals", action="store_true", help="document every module"
    )
    args = parser.parse_args()
    names = [f"scisynth.{name}" for name in PACKAGES]
    for name in names:
        module = importlib.import_module(name)
        if args.internals and hasattr(module, "__all__"):
            del module.__all__
    pdoc.render.configure(docformat="numpy")
    pdoc.pdoc(*names, PRIVATE, output_directory=OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
