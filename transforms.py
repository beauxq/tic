"""
There are 7 transformations (8 if you count the null transformation -
leaving the board as it is) that can be made on a tic-tac-toe board
without changing the state of the game.
This module defines an arbitrary preference among those transformations
so the neural network only needs to learn one of them.
"""

from typing import List, Set, Tuple, TypeVar
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

T = TypeVar('T', np.ndarray, int)
def transform(board_or_space: T,
              transform_type: int) -> T:
    """ according to the transform number,
    transform all the values on a board
    or (overload) transform the index of 1 space """
    if isinstance(board_or_space, (np.ndarray, list)):
        return np.array([board_or_space[from_i] for from_i in transformations[transform_type]])
    return transformations[invert_transform[transform_type]][board_or_space]


def choose_transformation(board: np.ndarray) -> Tuple[int, np.ndarray]:
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

BoardType = TypeVar('BoardType', np.ndarray, List[int])
def equal_transformations(board: BoardType) -> List[int]:
    """ returns list of all transformations
    equally preferred to this one (7)
    not including this one
    (which transformations make the board look the same?) """
    base_preference = np.dot(board, weights)
    equals = []
    for i in range(7):
        this_board = transform(np.array(board), i)
        this_preference = np.dot(this_board, weights)
        if this_preference == base_preference:
            equals.append(i)
    return equals


def equal_indexes(board: BoardType, index: int) -> Set[int]:
    """ which indexes are equivalent to the given index
    in all equal transformations """
    e_t = equal_transformations(board)
    to_return = {index}
    for tran in e_t:
        to_return.add(transform(index, tran))
    return to_return


def _test():
    original = np.array([randrange(-1, 2) for _ in range(9)])
    # sets that were a problem for previous implementation
    # original = [-1, -1, 0, 0, -1, 0, 1, 0, -1]
    # original = [0, 1, 0, -1, 0, -1, 0, 1, 0]
    # original = [1, 0, -1, 0, 0, 0, -1, 0, 1]
    # test transform
    # print(transform(original, 4))
    assert (transform(original, 4) == transform(transform(original, 5), 6)).all()
    assert (transform(np.array([-1, 0, 0, 0, 0, 1, 0, -1, 0]), 3)
            == [0, -1, 0, 0, 0, 1, -1, 0, 0]).all()

    # test choose
    print(original)
    t, new = choose_transformation(original)
    print(t, new)
    for i in range(8):
        this_t, this_c = choose_transformation(transform(original, i))
        print(this_t, this_c)
        assert (this_c == new).all()

    assert equal_transformations([1, 0, 0, 0, 0, -1, 0, -1, 0]) == [0]
    assert equal_transformations([0, 0, 0, 0, -1, 0, 0, 0, 0]) == [0, 1, 2, 3, 4, 5, 6]

    assert equal_indexes([0, 0, 0, 0, -1, 0, 0, 0, 0], 2) == {0, 2, 6, 8}
    assert equal_indexes([1, 0, 0, 0, 0, -1, 0, -1, 0], 2) == {2, 6}


if __name__ == "__main__":
    _test()
