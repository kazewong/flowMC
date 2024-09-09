from flowMC.utils.debug import enable_debug_verbose, get_mode


def test_enable_debug_verbose():
    assert get_mode() is False
    enable_debug_verbose()
    assert get_mode() is True
