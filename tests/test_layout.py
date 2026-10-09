"""The package layout: who may import whom, and what each subpackage exports."""

import ast
import importlib
from pathlib import Path

import pytest

import scisynth

SRC = Path(scisynth.__file__).parent

# Each subpackage and the subpackages it may import. This is what keeps the
# imports acyclic, so any one of them can be imported first. ``viz`` is left
# out of the lists: it is imported lazily, inside methods, and checked below.
ALLOWED = {
    "core": set(),
    "latent": {"core"},
    "stages": {"core"},
    "samplers": {"core", "latent"},
    "observer": {"core", "latent", "samplers", "stages"},
    "projection": {"core", "observer"},
    "testing": {"core", "latent", "stages", "observer"},
    "io": {"core", "latent", "observer"},
    "adapters": {"core", "latent"},
    "viz": {"core", "latent", "samplers", "stages", "observer"},
}
# Files that import more than their subpackage may, and why. The demo signals are
# built with a sampler and a latent, but ``Stage.demo_observation`` imports the
# module only when it is called, so no import cycle can form.
EXCEPTIONS = {"stages/demo.py": {"latent", "samplers"}}
EXPORTING = ["core", "latent", "samplers", "stages", "observer"]


def imported_subpackages(path: Path) -> set[str]:
    """Return the scisynth subpackages that a source file imports (relative imports)."""
    parts = ["scisynth", *path.relative_to(SRC).with_suffix("").parts]
    package = parts[:-1]
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package[: len(package) - node.level + 1]
            target = [*base, *(node.module.split(".") if node.module else [])]
            if len(target) > 1:
                found.add(target[1])
    return found


def owner(path: Path) -> str:
    relative = path.relative_to(SRC)
    return relative.stem if len(relative.parts) == 1 else relative.parts[0]


def test_subpackages_only_import_the_ones_below_them() -> None:
    problems = []
    for path in sorted(SRC.rglob("*.py")):
        who = owner(path)
        if who in {"__init__"}:
            continue  # the top level re-exports from everywhere
        imported = imported_subpackages(path) - {who, "viz"}
        extra = (
            imported - ALLOWED[who] - EXCEPTIONS.get(str(path.relative_to(SRC)), set())
        )
        if extra:
            problems.append(f"{path.relative_to(SRC)} imports {sorted(extra)}")
    assert not problems, problems


def test_only_the_viz_modules_import_viz_at_the_top_of_a_file() -> None:
    """Importing ``scisynth`` must not import ``scisynth.viz`` (it is optional)."""
    problems = []
    for path in sorted(SRC.rglob("*.py")):
        if owner(path) in {"viz", "__init__"}:
            continue
        for node in ast.parse(path.read_text()).body:  # top level only
            if isinstance(node, ast.ImportFrom) and node.level:
                module = node.module or ""
                if module.split(".")[0] == "viz" or (
                    node.level > 1 and module.startswith("viz")
                ):
                    problems.append(str(path.relative_to(SRC)))
    assert not problems, problems


@pytest.mark.parametrize("name", EXPORTING)
def test_a_subpackage_exports_its_public_names(name: str) -> None:
    module = importlib.import_module(f"scisynth.{name}")
    assert len(module.__all__) == len(set(module.__all__))
    for exported in module.__all__:
        assert getattr(module, exported).__module__.startswith("scisynth"), exported


def test_the_top_level_repeats_the_subpackage_names() -> None:
    seen: dict[str, str] = {}
    for name in EXPORTING:
        module = importlib.import_module(f"scisynth.{name}")
        for exported in module.__all__:
            assert exported not in seen, f"{exported} is in {seen[exported]} and {name}"
            seen[exported] = name
            assert getattr(scisynth, exported) is getattr(module, exported)
    # everything at the top level comes from a subpackage, except these
    top = set(scisynth.__all__) - set(seen)
    assert top == {"Projection", "__version__"}
