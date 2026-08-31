from pararnn import __version__


def test_version_is_set():
    assert __version__
    assert __version__ != "0.0.0"
