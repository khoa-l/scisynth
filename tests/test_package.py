import scisynth


def test_version_and_explicit_exports() -> None:
    assert isinstance(scisynth.__version__, str)
    assert all(hasattr(scisynth, name) for name in scisynth.__all__)
