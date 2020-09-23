from typing import List, Union
from random import randrange
import numpy as np

transformations = (
    # flip on \
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
    # flip on /
    (8, 5, 2, 7, 4, 1, 6, 3, 0),
    # flip on |
    (2, 1, 0, 5, 4, 3, 8, 7, 6),
    # flip on -
    (6, 7, 8, 3, 4, 5, 0, 1, 2),
    # rotate 90 degrees left
    (2, 5, 8, 1, 4, 7, 0, 3, 6),
    # rotate 180 degrees
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
    # rotate 90 degrees right
    (6, 3, 0, 7, 4, 1, 8, 5, 2),
    # rotate 0 degrees
    (0, 1, 2, 3, 4, 5, 6, 7, 8)
)

# transformation required to return to original
invert_transform = (0, 1, 2, 3, 6, 5, 4, 7)

# to give a preferred transformation
weights = tuple(3**i for i in range(1, 10))


def transform(board_or_space: Union[np.ndarray, int], transform_type: int) -> Union[np.ndarray, int]:
    """ according to the transform number,
    transform all the values on a board
    or transform the index of 1 space """
    if isinstance(board_or_space, np.ndarray):
        return np.array([board_or_space[from_i] for from_i in transformations[transform_type]])
    return transformations[invert_transform[transform_type]][board_or_space]


def choose_transformation(board: np.ndarray) -> (int, List[int]):
    """ returns a tuple of
    (which transform is preferred (int), that transformation (board)) """
    min_t = 7
    min_value = np.dot(board, weights)
    min_board = board
    for i in range(7):
        this_board = transform(board, i)
        this_value = np.dot(this_board, weights)
        if this_value < min_value:
            min_value = this_value
            min_t = i
            min_board = this_board
    return min_t, min_board


def test():
    original = np.array([randrange(-1, 2) for _ in range(9)])
    # sets that were a problem for previous implementation
    # original = [-1, -1, 0, 0, -1, 0, 1, 0, -1]
    # original = [0, 1, 0, -1, 0, -1, 0, 1, 0]
    # original = [1, 0, -1, 0, 0, 0, -1, 0, 1]
    # test transform
    # print(transform(original, 4))
    assert transform(original, 4) == transform(transform(original, 5), 6)
    assert transform(np.array([-1, 0, 0, 0, 0, 1, 0, -1, 0]), 3) == [0, -1, 0, 0, 0, 1, -1, 0, 0]

    # test choose
    print(original)
    t, new = choose_transformation(original)
    print(t, new)
    for i in range(8):
        this_t, this_c = choose_transformation(transform(original, i))
        print(this_t, this_c)
        assert this_c == new


if __name__ == "__main__":
    test()
