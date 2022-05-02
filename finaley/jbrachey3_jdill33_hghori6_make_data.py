import pandas as pd
import os
import torch
import chess.pgn
import numpy as np
import csv
from typing import List

# device = ("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# NOTE ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~
# To instantiate data for running neural net just run this file
# ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~
pgn = open("data/data.pgn")


total_positions = 500000

# load board position numpy matrix from csv file given index
def load_position(idx):
    position_shape = (9, 8, 8)
    # retrieving data from file.
    pos = np.loadtxt("data/board_positions/board"+str(idx)+'.csv')
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loaded_pos = pos.reshape(pos.shape[0], pos.shape[1] // position_shape[2], position_shape[2])
    # print(loaded_pos)
    return loaded_pos
def store_position(pos, idx):
    with open("data/board_positions/board"+str(idx)+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
    # reshaping the array from 3D matrice to 2D matrice.
    arrReshaped = pos.reshape(pos.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt("data/board_positions/board"+str(idx)+'.csv', arrReshaped)

def convert_pos_to_bitboard(pos):
    board = chess.Board()
    board.clear_board()
    b_pawn = chess.Piece(1, False)
    w_pawn = chess.Piece(1, True)
    b_knight = chess.Piece(2, False)
    w_knight = chess.Piece(2, True)
    b_bishop = chess.Piece(3, False)
    w_bishop = chess.Piece(3, True)
    b_rook = chess.Piece(4, False)
    w_rook = chess.Piece(4, True)

    b_queen = chess.Piece(5, False)
    w_queen = chess.Piece(5, True)

    b_king = chess.Piece(6, False)
    w_king = chess.Piece(6, True)

    # piece_mat = np.empty((6, 2))
    # rows, cols = (6, 2)
    # piece_mat = [[0]*cols]*rows
    # # print(arr)
    # for p in range(6):
    #     piece_mat[p][0] = chess.Piece(p + 1, True)
    #     piece_mat[p][1] = chess.Piece(p + 1, False)
    for row in range(8):
        for col in range(8):
            # for p in range(6):
            #     coordinate = ((7 - row) * 8) + col
            #     if pos[p,row,col] == -1:
            #         board.set_piece_at(coordinate, piece_mat[p][1])
            #     elif pos[p,row,col] == 1:
            #         board.set_piece_at(coordinate, piece_mat[p][0])
            # pawns
            if pos[0,row,col] == -1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, b_pawn)
            elif pos[0,row,col] == 1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, w_pawn)
            # knights
            if pos[1,row,col] == -1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, b_knight)
            elif pos[1,row,col] == 1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, w_knight)
            # bishop
            if pos[2,row,col] == -1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, b_bishop)
            elif pos[2,row,col] == 1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, w_bishop)
            # rook
            if pos[3,row,col] == -1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, b_rook)
            elif pos[3,row,col] == 1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, w_rook)
            # queen
            if pos[4,row,col] == -1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, b_queen)
            elif pos[4,row,col] == 1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, w_queen)
            # king
            if pos[5,row,col] == -1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, b_king)
            elif pos[5,row,col] == 1:
                coordinate = ((7 - row) * 8) + col
                board.set_piece_at(coordinate, w_king)
    # print("converted back:, \n", board)
    return board
def parse_game():
    cur_game = chess.pgn.read_game(pgn)
    board = cur_game.board()
    # i = 0
    # position = parse_board(board)
    # store_position(position)
    # for move in cur_game.mainline_moves():
    #     parse_board(board)
    #     board.push(move)
    #     i += 1

    return cur_game

def parse_board(board):
    position = np.zeros((9, 8, 8)) 
    # first 2 dimensions the dimension of board
    # 3rd channel 0-6 is each piece type
    # 3rd channel 7-9 is extra data, color to move, en pessant, castling rights
    # cur_game = chess.pgn.read_game(pgn)
    # board = cur_game.board()

    for i in range(0, 6):
        cur_piece_w = board.pieces(i + 1, True)
        cur_piece_b = board.pieces(i + 1, False)
        for piece in cur_piece_w:
            col = piece % 8
            row =  7 - int(piece / 8) 
            position[i, row, col] = 1
        for piece in cur_piece_b:
            col = piece % 8
            row =  7 - int(piece / 8) 
            position[i, row, col] = -1

    position[6, :, :] = np.zeros((8,8))
    if board.has_legal_en_passant():
        square = board.ep_square
        col = square % 8
        row =  7 - int(square / 8) 
        position[row, col, 6] = 1
        if not board.turn: # black to play
            position[6, :, :] *= -1 
            position[8, :, :] = np.ones((8, 8)) * -1 # for setting color feature 
        else:
            position[8, :, :] = np.ones((8, 8))


    castle_mat = np.zeros((8,8))
    if bool(board.castling_rights & chess.BB_H1):
        castle_mat[4:, 4:] = np.ones((4,4))
    if bool(board.castling_rights & chess.BB_A1):
        castle_mat[4:, 0:4] = np.ones((4,4))
    if bool(board.castling_rights & chess.BB_A8):
        castle_mat[0:4, 0:4] = np.ones((4,4)) * -1
    if bool(board.castling_rights & chess.BB_H8):
        castle_mat[0:4, 4:] = np.ones((4,4)) * -1
    position[7, :, :] = castle_mat

    # print(position)
    return position


# make stockfish value file
file = open("data/stockfish.csv")
reader =csv.reader(file)
header = next(reader)
print(header)

cur_idx = 0
breakout = False
with open('data/stockfish_modified.TXT', 'w') as f:
    for row in reader:
        if breakout:
            break
        game_scores = row[1].split()
        for score in game_scores:
            if cur_idx >= total_positions: 
                breakout = True
                break
            f.write(score)
            f.write('\n')
            cur_idx += 1
            #     print(score)
            # f.flush()
    # print(rows)

# write board positions

# with open('data/stockfish_modified.TXT') as f:
#     print(f.read())
i = 0
while i < total_positions:
    cur_game = chess.pgn.read_game(pgn)
    board = cur_game.board()
    for move in cur_game.mainline_moves():
        if i >= total_positions:
            break
        board_pos = parse_board(board)
        store_position(board_pos, i)
        board.push(move)
        i += 1
    if i >= total_positions:
        break
# load_position(15)    
# load_position(36)
# pos = torch.from_numpy(load_position(37))
# print(pos)
# convert_pos_to_bitboard(pos)

# with open('data/stockfish_modified.TXT') as f:
#     file_contents = f.readlines()
#     eval =  file_contents[49999]
#     print(eval)
