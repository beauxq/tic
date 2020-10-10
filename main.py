from typing import List, Tuple
from random import random, choice

import numpy as np

from transforms import choose_transformation, invert_transform, transform, equal_indexes
from winning import get_valid_moves, check_win, get_winning_moves, can_block
from layer import Layer
from network import Network


def log(*out):
    """ print, only if static logging is true """
    if log.logging:
        if (len(out) == 1) \
                and isinstance(out[0], (list, np.ndarray)) \
                and len(out[0]) == 9:
            # guess it's a tic tac toe board
            for i, value in enumerate(out[0]):
                if isinstance(value, float):
                    value = round(value, 3)
                elif value > -1:
                    value = " " + str(value)
                print(value, end=("\n" if i % 3 == 2 else " "))
            print()
        else:
            print(*out)

log.logging = False


def get_ai_move(board: List[int], player: int, tic_net: Network, randomness: float) -> int:
    """ return which space to put player's mark in """
    # copy board to not modify original
    board_copy = np.copy(board)
    # make it so I am player 1
    board_copy *= player
    # work with preferred transformation
    transform_used, preferred_transformation = choose_transformation(board_copy)
    log("preferred transform:")
    log(preferred_transformation)

    outputs = tic_net.predict(np.array([preferred_transformation]))
    log(outputs[0])
    valid_moves = get_valid_moves(preferred_transformation)
    max_move = next(iter(valid_moves))
    for valid_move in valid_moves:
        if outputs[0][valid_move] > outputs[0][max_move]:
            max_move = valid_move

    prob_of_using_max_move = (outputs[0][max_move] / (randomness + 0.0000001) - 1)
    if random() < 1 - prob_of_using_max_move:
        log("made a random choice from probability", 1 - prob_of_using_max_move)
        return choice(tuple(get_valid_moves(board)))

    # now transform max_move back to original board
    return transform(max_move, invert_transform[transform_used])


def get_human_move(board):
    log_temp = log.logging
    log.logging = True

    log(board)
    answer = -1
    while not (isinstance(answer, int) and -1 < answer < 9):
        try:
            answer = int(input("space? [0-8] "))
            if board[answer] != 0:
                raise "no"
        except:
            pass

    log.logging = log_temp
    return answer


def play_a_game(tic_net: Network, with_training: bool, amount_of_randomness: float, human: int):
    """ Play a game, neural network against itself.

    `with_training` to train `tic_net` on data gathered from this game
    
    `amount_of_randomness` [0, 1] to choose moves

    `human` is 1 or -1 or 0, for first, second, or no human """
    board = [0 for _ in range(9)]
    move_record: List[Tuple[List[int], int]] = []  # list of (board, move)
    winner = 0
    while True:
        # player 1
        if human == 1:
            space = get_human_move(board)
        else:
            space = get_ai_move(board, 1, tic_net, amount_of_randomness)
        move_record.append((np.copy(board), space))
        board[space] = 1
        log(board)
        if check_win(board, 1):
            winner = 1
            break
        # see if board is full (player 1 should be last)
        if len(get_valid_moves(board)) == 0:
            break
        # player 2
        if human == -1:
            space = get_human_move(board)
        else:
            space = get_ai_move(board, -1, tic_net, amount_of_randomness)
        move_record.append((np.copy(board), space))
        board[space] = -1
        log(board)
        if check_win(board, -1):
            winner = -1
            break
    log("winner is", winner)

    if with_training:
        train(move_record, winner, tic_net)


def train(move_record: List[Tuple[List[int], int]], winner: int, tic_net: Network):
    """ train network based on outcome """
    training_sets = []
    target_output = []
    player = 1
    for board, moved in move_record:
        valid_moves = get_valid_moves(board)
        tie_value = 0.7  # target output for training data leading to tie
        invalid_value = 0.5  # target output for training data of invalid move
        # for moves I could have taken but didn't
        default_value = 0.3 if player == winner else (0.5 if winner == (-1 * player) else 0.5)
        # TODO: maybe, instead of 0.5, use the value that the net already predicts for that space?
        out = [(default_value if v == 0 else invalid_value) for v in board]

        equals = equal_indexes(board, moved)
        for i in equals:
            # don't know what values I should use if I'm not sure how good of a move this is
            out[i] = 0.9 if winner == player else (0.1 if winner == (-1 * player) else tie_value)

        # overwrite that if there are winning moves or blocking moves or only 1 possible move
        winning_moves = get_winning_moves(board, player)
        if len(winning_moves) > 0:
            out = [1 if i in winning_moves else (0 if (i in valid_moves) else invalid_value)
                   for i in range(9)]
        else:  # no winning moves
            can, blocking_move = can_block(board, player)
            if can:
                out = [1 if i == blocking_move else (0 if (i in valid_moves) else invalid_value)
                       for i in range(9)]
            else:  # no wining moves and no blocking moves
                if len(valid_moves) == 1:
                    log("only 1 valid on this board")
                    out[next(iter(valid_moves))] = 1

        board_copy = np.copy(board)
        board_copy *= player
        transform_used, preferred_transform_of_board = choose_transformation(board_copy)
        transformed_output = transform(np.array(out), transform_used)

        log("training:")
        log(preferred_transform_of_board)
        log(transformed_output)
        training_sets.append(preferred_transform_of_board)
        target_output.append(transformed_output)

        player *= -1

    tic_net.train(np.array(training_sets), np.array(target_output), 1, 0.0625, log.logging)


def play_with_human(tic_net):
    answer = input("you want to go first? (y/n)").lower()
    if (answer and (answer[0] == "n")):
        player = -1
    else:
        player = 1
    print("you are", player)
    play_a_game(tic_net, False, 0, player)


def main():
    hidden_activation = Layer.TruncatedSQRT
    tic_net = Network(9)
    tic_net.add_layer(30, hidden_activation)
    tic_net.add_layer(30, hidden_activation)
    tic_net.add_layer(30, hidden_activation)
    tic_net.add_layer(9, Layer.Sigmoid)

    game_count = 60000
    for game in range(game_count):
        log.logging = ((game % 1000 == 0) or (game > (game_count - 5)))
        log("game:", game)
        amount_of_randomness = (1 - (0.75 * game / game_count))
        log("randomness:", amount_of_randomness)
        play_a_game(tic_net, True, amount_of_randomness, 0)


    log.logging = True
    play_a_game(tic_net, False, 0, 0)

    # human
    answer = input("play? (y/n) ").lower()
    while (not answer) or (answer[0] != "n"):
        play_with_human(tic_net)
        answer = input("again? (y/n) ").lower()
    
    # log(tic_net)


if __name__ == "__main__":
    main()
