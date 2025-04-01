def test_import():
    import imagewand

    assert imagewand.__name__ == "imagewand"


def test_cli_import():
    from imagewand import cli

    assert cli.__name__ == "imagewand.cli"
