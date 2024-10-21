from src.utilities import is_tuple_of_ints, is_tuple_of_pairs


def test_is_tuple_of_ints() -> None:
    v1 = (0, 1, 2, 3, 4)
    assert is_tuple_of_ints(v1) is True
    v2 = ((0, 0), (1, 2), (3, 4))
    assert is_tuple_of_ints(v2) is False


def test_is_tuple_of_pairs() -> None:
    v1 = (0, 1, 2, 3, 4)
    assert is_tuple_of_pairs(v1) is False
    v2 = ((0, 0), (1, 2), (3, 4))
    assert is_tuple_of_pairs(v2) is True
