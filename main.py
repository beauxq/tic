from typing import List, Tuple
from random import random, choice

import numpy as np

from transforms import choose_transformation, invert_transform, transform
from winning import get_valid_moves, check_win, get_winning_moves, can_block
from layer import Layer
from network import Network


def log(*out):
    if log.logging:
        if (len(out) == 1) \
                and isinstance(out[0], (list, np.ndarray)) \
                and len(out[0]) == 9:
            # guess it's a tic tac toe board
            for i, v in enumerate(out[0]):
                print(v, end=("\n" if i % 3 == 2 else " "))
            print()
        else:
            print(*out)


log.logging = False


def move(board: List[int], player: int, tic_net: Network) -> int:
    """ return which space to put player's mark in """
    # copy board to not modify original
    board_copy = np.array(board[:])
    # make it so I am player 1
    board_copy *= player
    # work with preferred transformation
    transform_used, preferred_transformation = choose_transformation(board_copy)
    log("preferred transform:")
    log(preferred_transformation)

    outputs = tic_net.predict(np.array([preferred_transformation]))
    log(outputs[0])
    valid_moves = get_valid_moves(preferred_transformation)
    max_move = valid_moves[0]
    for valid_move in valid_moves:
        if outputs[0][valid_move] > outputs[0][max_move]:
            max_move = valid_move

    if random() < 1 - outputs[0][max_move]:
        log("made a random choice from probability", 1 - outputs[0][max_move])
        return choice(get_valid_moves(board))

    # now transform max_move back to original board
    return transform(max_move, invert_transform[transform_used])


def play_a_game(tic_net: Network):
    board = [0 for _ in range(9)]
    move_record: List[Tuple[List[int], int]] = []  # list of (board, move)
    winner = 0
    while True:
        # player 1
        a = move(board, 1, tic_net)
        move_record.append((board[:], a))
        board[a] = 1
        log(board)
        if check_win(board, 1):
            winner = 1
            break
        # see if board is full (player 1 should be last)
        if len(get_valid_moves(board)) == 0:
            break
        # player 2
        a = move(board, -1, tic_net)
        move_record.append((board[:], a))
        board[a] = -1
        log(board)
        if check_win(board, -1):
            winner = -1
            break
    log("winner is", winner)

    # train network based on outcome
    training_sets = []
    target_output = []
    player = 1
    for board, moved in move_record:
        # TODO: maybe, instead of 0.5, use the value that the net already predicts for that space?
        out = [(0.5 if v == 0 else 0) for v in board]  # default: all the moves I didn't make are neutral
        # don't know what values I should use if I'm not sure how good of a move this is
        out[moved] = 0.9 if winner == player else (0.1 if winner == -1 * player else 0.5)

        # overwrite that if there are winning moves or blocking moves
        winning_moves = get_winning_moves(board, player)
        if len(winning_moves) > 0:
            out = [1 if i in winning_moves else 0 for i in range(9)]
        else:  # no winning moves
            can, blocking_move = can_block(board, player)
            if can:
                out = [1 if i == blocking_move else 0 for i in range(9)]

        board_copy = np.array(board[:])
        board_copy *= player
        transform_used, preferred_transform_of_board = choose_transformation(board_copy)
        transformed_output = transform(np.array(out), transform_used)

        log("training:")
        log(preferred_transform_of_board)
        log(transformed_output)
        training_sets.append(preferred_transform_of_board)
        target_output.append(transformed_output)

        player *= -1

    tic_net.train(np.array(training_sets), np.array(target_output), 1, 0.0625, int(log.logging))


def main():
    tic_net = Network(9)
    tic_net.add_layer(30, Layer.Sigmoid)
    tic_net.add_layer(30, Layer.Sigmoid)
    tic_net.add_layer(30, Layer.Sigmoid)
    tic_net.add_layer(9, Layer.Sigmoid)

    for game in range(50000):
        log.logging = ((game % 1000 == 0) or (game > 49995))
        log("game:", game)
        play_a_game(tic_net)


if __name__ == "__main__":
    main()
