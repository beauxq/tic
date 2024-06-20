""" utility functions related to plays in the game of tic-tac-toe """

from typing import TYPE_CHECKING, List, Set, Tuple
import numpy as np

from transforms import BoardType

if TYPE_CHECKING:
    from _typeshed import SupportsGetItem

possible_win_sets = (
    # horizontal lines
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    # vertical lines
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    # diagonal lines
    (0, 4, 8),
    (2, 4, 6)
)


def check_win(board: "SupportsGetItem[int, float]", player: int) -> bool:
    """ player is 1 or -1 """
    target = 3 * player
    for win in possible_win_sets:
        total = 0
        for i in win:
            total += board[i]
        if total == target:
            return True
    return False


def get_valid_moves(board: BoardType) -> Set[int]:
    """ places not already taken by a player """
    return {i for i, v in enumerate(board) if v == 0}


def get_winning_moves(board: BoardType, player: int) -> List[int]:
    """ indexes where player could go to win
        - player is 1 or -1 """
    to_return: list[int] = []
    empty_spaces = get_valid_moves(board)
    for space_i in empty_spaces:
        new_board = np.copy(board)
        new_board[space_i] = player
        if check_win(new_board, player):
            to_return.append(space_i)
    return to_return


def can_block(board: BoardType, player: int) -> Tuple[bool, int]:
    """ returns whether player can make a move
    to block the other from winning
    and what space that move is in """
    other = -1 * player
    winning_moves = get_winning_moves(board, other)
    return len(winning_moves) == 1, winning_moves[0] if len(winning_moves) == 1 else -1


def _test():
    assert check_win([0, 1, 0, 0, 1, 0, 0, 1, 0], 1)
    assert not check_win([0, 1, 0, 0, -1, 0, 0, 1, 0], 1)
    assert check_win([0, 1, -1, 0, -1, 0, -1, 1, 0], -1)
    assert not check_win([-1, 1, 0, 0, -1, -1, -1, -1, 0], -1)

    assert get_winning_moves([0, 1, 0, 0, 0, 1, 1, 0, 0], 1) == []
    assert get_winning_moves([0, 1, 0, 0, 1, 0, 0, 0, 0], 1) == [7]
    assert get_winning_moves([0, 1, 0, 1, 1, 0, 0, 0, 0], 1) == [5, 7]
    assert get_winning_moves([1, 0, 0, 0, 1, 0, 1, 1, 0], 1) == [1, 2, 3, 8]
    assert get_winning_moves([0, 1, 0, 1, 0, 1, 0, 1, 0], 1) == [4]

    assert can_block([0, 1, 0, 0, 0, 1, 1, 0, 0], -1) == (False, -1)
    assert can_block([0, 1, 0, 0, 1, 0, 0, 0, 0], -1) == (True, 7)
    assert can_block([0, 1, 0, 1, 1, 0, 0, 0, 0], -1) == (False, -1)
    assert can_block([1, 0, 0, 0, 1, 0, 1, 1, 0], -1) == (False, -1)
    assert can_block([0, 1, 0, 1, 0, 1, 0, 1, 0], -1) == (True, 4)


if __name__ == "__main__":
    _test()
