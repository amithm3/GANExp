def _assert_positive_integer(**params):
    for name, value in params.items():
        assert isinstance(value, int) and value > 0, \
            f"Invalid value of {name}, must be a positive integer, got {value}"


def _assert_integer(**params):
    for name, value in params.items():
        assert isinstance(value, int), \
            f"Invalid value of {name}, must be an integer, got {value}"
