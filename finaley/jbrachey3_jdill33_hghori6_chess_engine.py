import chess.pgn
import chess
import math
import numpy as np
from jbrachey3_jdill33_hghori6_ChessCNN import ChessCNN
import torch

# import sys
# sys.setrecursionlimit(5000)

# material table for pawn(0) - queen(4):
material_table = np.asarray([82, 397, 320, 420, 1025])

# N/A, pawn, knight, bishop, rook, queen
piece_attack_weights = [0, 17, 35, 30, 55, 110]

# 0 - 7 attacking pieces (not including pawns)
num_piece_attackers_weights = [0, 35, 65, 85, 95, 97, 99, 100]

# king saftey table
def king_attack_score(enemyKingColor, board):
    kingPos = board.king(enemyKingColor)
    # print(kingPos)
      # convert king pos into matrix coord
    row = 7 - int(kingPos/8)
    col = kingPos % 8
    score = 0 # total value of each attack piece type
    num_attackers = 0
    if row + 1 < 8: # below king
        piece = board.piece_at(kingPos - 8)
        if piece != None and piece.color != enemyKingColor and piece.piece_type != 6:
            score += piece_attack_weights[piece.piece_type]
            if piece.piece_type != 1:
                num_attackers += 1
    if row - 1 >= 0: #above king
        piece = board.piece_at(kingPos + 8)
        if piece != None and piece.color != enemyKingColor and piece.piece_type != 6:
            score += piece_attack_weights[piece.piece_type]
            if piece.piece_type != 1:
                num_attackers += 1
    if col + 1 < 8: # right of king
        piece = board.piece_at(kingPos + 1)
        if piece != None and piece.color != enemyKingColor and piece.piece_type != 6:
            score += piece_attack_weights[piece.piece_type]
            if piece.piece_type != 1:
                num_attackers += 1
    if col - 1 >= 0: # left of king
        piece = board.piece_at(kingPos - 1)
        if piece != None and piece.color != enemyKingColor and piece.piece_type != 6:
            score += piece_attack_weights[piece.piece_type]
            if piece.piece_type != 1:
                num_attackers += 1
    # print("read",enemyKingColor)
  # zone in front of king
  # if enemy king is white, rows above king are the zone (lower rows)
    if enemyKingColor:
        for c in range(col - 1, col + 2):
            for r in range(row - 3, row + 1):
                # print('c', c)
                # print('r', r)
                if 0 <= c < 8 and 0 <= r < 8:
                    try:
                        # print(piece)
                        # print(piece.color)
                        p = 9
                    except AttributeError:
                        i = 0
                    piece = board.piece_at( ((7 - r) * 8) + c)
                    if piece != None and piece.color != enemyKingColor and piece.piece_type != 6:
                        score += piece_attack_weights[piece.piece_type]
                        if piece.piece_type != 1:
                            num_attackers += 1
    else: #higher rows enemy zone for black king
        for c in range(col - 1, col + 2):
            for r in range(row, row + 3):
                # print("c: ", c)
                # print("r: ", r)
                if 0 <= c < 8 and 0 <= r < 8:
                    piece = board.piece_at( ((7 - r) * 8) + c)
                    try:
                        # print(piece)
                        # print(piece.color)
                        p = 9
                    except AttributeError:
                        i = 0
                    if piece != None and piece.color != enemyKingColor and piece.piece_type < 6:
                        # print(piece)
                        score += piece_attack_weights[piece.piece_type]
                        if piece.piece_type != 1:
                            num_attackers += 1
    num_attackers = min(7, num_attackers)
    king_attack_score = score * (num_piece_attackers_weights[num_attackers] / 100)
    return king_attack_score
# king_attack_score(True, chess.Board())

def black_piece_table(table):
    new_table = np.flip(table, 0) 
    return new_table

# opening_bonus
bonus = 55
opening_moves = 7
#opening piece tables:
opening_pawn_table = np.asarray([
                          [0,  0,  0,  0,  0,  0,  0,  0],
                          [98, 134,  61,  95,  68, 126, 74, 70],
                          [10, 10, 20, 30, 30, 30, 15, 15],
                          [5,  5, 10, 25, 25, 10,  5,  5],
                          [0,  0, 10, 20, 20 + bonus,  0,  0,  0],
                          [5,  0, 10, 10 + bonus, 10,-10, -10, 0],
                          [5, 10, 10,-20,-20, 10, 10,  5],
                          [0,  0,  0,  0,  0,  0,  0,  0]
                         ])
opening_knight_table = np.asarray([
                           [ -75,  -50,  -10,  5,  5,  5,  5,  5],
                           [ -60,  0,  5,  10, 10, 5,  5,  5],
                           [-45,-5,  10, 20, 20, 10, 5,  5],
                           [-15,-5,  10, 10, 10, 10, 0,  0],
                           [-15, 5,  15, 30, 30, 15, 5,  0],
                           [-30, 3,  15, 20, 20, 15 + bonus * 2, 5, 0],
                           [-30, -20,  0, 10, 10, 0, 0, 0],
                           [-50, -40, -30, -20, -5, 0, 0, 0]
                          ])
opening_bishop_table = np.asarray([
                           [-20,-10,-10,-10,-10, 0,0,0],
                           [-10,  0,  0,  0,  0,  1,  1, 1],
                           [-10,  0,  5, 10, 10,  2,  2,2],
                           [-10,  5,  5, 10, 10,  3,  3,-10],
                           [-10,  0, 10, 10, 10, 10,  7,-10],
                           [-10, 10, 10, 5, 10, 10, 10,-10],
                           [-10,  5,  0,  0,  0 + bonus,  0,  5,-10],
                           [-20,-10,-10,-10,-10,-10,-10,-20]
                          ])
opening_king_table = np.asarray([
                          [-165,  -146,  -146, -115, -156, -134,   -146,  -146],
                          [-146,  -110, -120,  -115,  -115,  -146, -138, -129],
                          [-115,  -115, -115, -100, -100,   -115, -120, -100],
                          [-55, -70, -50, -50, -60, -65, -54, -56],
                          [-49,  -36, -27, -39, -46, -44, -33, -51],
                          [-14, -14, -22, -46, -44, -30, -15, -27],
                          [1,   7,  -8, -64, -43, -16,   9,   8],
                          [-15,  36,  12, -54,   8, -28,  24 + bonus,  14]
                         ]) * 10
# piece tables: 
# note for black side to mirror the array on the y axis
pawn_table = np.asarray([
                          [0,  0,  0,  0,  0,  0,  0,  0],
                          [98, 134,  61,  95,  68, 126, 74, 70],
                          [10, 10, 20, 30, 30, 30, 15, 15],
                          [5,  5, 10, 25, 25, 10,  5,  5],
                          [0,  0, 10, 20, 20,  0,  0,  0],
                          [5,  0, 10, 20, 10,-10, -10, 0],
                          [5, 10, 10,-20,-20, 10, 10,  5],
                          [0,  0,  0,  0,  0,  0,  0,  0]
                         ])
knight_table = np.asarray([
                           [ -75,  -50,  -10,  5,  5,  5,  5,  5],
                           [ -60,  0,  5,  10, 10, 5,  5,  5],
                           [-45,-5,  10, 20, 20, 10, 5,  5],
                           [-15,-5,  10, 30, 30, 10, 0,  0],
                           [-15, 5,  15, 30, 30, 15, 5,  0],
                           [-30, 3,  15, 20, 20, 15, 5, 0],
                           [-30, -20,  0, 10, 10, 0, 0, 0],
                           [-50, -40, -30, -20, -5, 0, 0, 0]
                          ])
bishop_table = np.asarray([
                           [-20,-10,-10,-10,-10, 2,2,2],
                           [-10,  0,  0,  0,  0,  3,  3, 3],
                           [-20,  0,  5, 10, 10,  5,  5,5],
                           [-10,  0,  5, 10, 10,  5,  0,-10],
                           [-10,  0, 15, 10, 10, 10,  0,-10],
                           [-10, 10, 10, 5, 10, 10, 10,-10],
                           [-10,  5,  0,  0,  0,  0,  5,-10],
                           [-20,-10,-10,-10,-10,-10,-10,-20]
                          ])
rook_table = np.asarray([
                          [32,  42,  32,  51, 63,  20,  51,  43],
                          [27,  32,  58,  62, 80, 67,  56,  44],
                          [-5,  19,  26,  36, 17, 45,  61,  30,],
                          [-24, -11,   7,  26, 24, 35,  -8, -20],
                          [-36, -26, -12,  -1,  9, -7,   6, -23],
                          [-45, -25, -16, -17,  3,  0,  -5, -33],
                          [-44, -16, -20,  -9, -1, 11,  -6, -71],
                          [-19, -13,   1,  17, 16,  7, -37, -26]
                        ])
queen_table = np.asarray([
                          [-28,   0,  29,  12,  59,  44,  43,  45],
                          [-24, -39,  -5,   1, -16,  57,  28,  54],
                          [-13, -17,   7,   8,  29,  56,  47,  57],
                          [-27, -27, -16, -16,  -1,  17,  -2,   1],
                          [-9, -26,  -9, -10,  -2,  -4,   3,  -3],
                          [-14,   2, -11,  -2,  -5,   2,  14,   5],
                          [-35,  -8,  11,   2,   8,  15,  -3,   1],
                          [-1, -18,  -9,  10, -15, -25, -31, -50],   
                         ])
king_table = np.asarray([
                          [-165,  -146,  -146, -115, -156, -134,   -146,  -146],
                          [-146,  -110, -120,  -115,  -115,  -146, -138, -129],
                          [-115,  -115, -115, -100, -100,   -115, -120, -100],
                          [-55, -70, -50, -50, -60, -65, -54, -56],
                          [-49,  -36, -27, -39, -46, -44, -33, -51],
                          [-14, -14, -22, -46, -44, -30, -15, -27],
                          [1,   7,  -8, -64, -43, -16,   9,   8],
                          [-15,  36,  12, -54,   8, -28,  24,  14]
                         ]) * 10

# print(black_piece_table(pawn_table))

# convert board info for NN input
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
    # return torch.from_numpy(position).type(torch.FloatTensor)
    return position

def nn_eval(board):
    nn_model = ChessCNN()
    model_state_dict = torch.load("model_params.pt")
    #model_state_dict = torch.load("model_params/model_params.pt", map_location=torch.device('cpu'))
    nn_model.load_state_dict(model_state_dict["model_state_dict"])
    input = parse_board(board)
    input = np.asarray([input]) # this solves my issue of feedforward?!!!!!!!
    # print(input)
    input = torch.from_numpy(input).type(torch.FloatTensor)
    output = nn_model(input)
    # print(output.size())
    return float(output)

def material_eval(board):
    black_score = 0
    white_score = 0
    for i in range(0, 5): #0 - 4 corresponding to pawn - queen
        white_pieces = board.pieces(piece_type = i + 1, color = True)
        black_pieces = board.pieces(piece_type = i + 1, color = False)

        white_score += len(white_pieces) * material_table[i]
        black_score += len(black_pieces) * material_table[i]
    return white_score - black_score

def pos_to_row_col(board_pos):
    row = 7 - int(board_pos/8)
    col = board_pos % 8
    return (row, col)
def piece_table_eval(board):
    in_opening = False
    if board.fullmove_number < opening_moves:
        in_opening = True
    black_score = 0
    white_score = 0
    for i in range(0, 6): #0 - 5 corresponding to pawn - king
        white_pieces = board.pieces(piece_type = i + 1, color = True)
        black_pieces = board.pieces(piece_type = i + 1, color = False)
        for piece_coord in white_pieces:
            row, col = pos_to_row_col(piece_coord)
            if i == 0:
                if in_opening:
                    white_score += opening_pawn_table[row,col]
                else: 
                    white_score += pawn_table[row,col]
            elif i == 1:
                if in_opening:
                    white_score += opening_knight_table[row,col]
                else: 
                    white_score += knight_table[row,col]
            elif i == 2:
                if in_opening:
                    white_score += opening_bishop_table[row,col]
                else: 
                    white_score += bishop_table[row,col]
            elif i == 3:
                white_score += rook_table[row,col]
            elif i == 4:
                white_score += queen_table[row,col]
            elif i == 5:
                if in_opening:
                    white_score += opening_king_table[row,col]
                else: 
                    white_score += king_table[row,col]

        for piece_coord in black_pieces:
            row, col = pos_to_row_col(piece_coord)
            if i == 0:
                if in_opening:
                    black_score += black_piece_table(opening_pawn_table)[row,col]
                else: 
                    black_score += black_piece_table(pawn_table)[row,col]
                black_score += black_piece_table(pawn_table)[row,col]
            elif i == 1:
                if in_opening:
                    black_score += black_piece_table(opening_knight_table)[row,col]
                else: 
                    black_score += black_piece_table(knight_table)[row,col]
            elif i == 2:
                if in_opening:
                    black_score += black_piece_table(opening_bishop_table)[row,col]
                else: 
                    black_score += black_piece_table(bishop_table)[row,col]
            elif i == 3:
                black_score += black_piece_table(rook_table)[row,col]
            elif i == 4:
                black_score += black_piece_table(queen_table)[row,col]
            elif i == 5:
                if in_opening:
                    black_score += black_piece_table(opening_king_table)[row,col]
                else: 
                    black_score += black_piece_table(king_table)[row,col]
    return white_score - black_score
def king_attack_eval(board):
    white_score = king_attack_score(enemyKingColor = False, board = board)
    black_score = king_attack_score(enemyKingColor = True, board = board)
    # print("white: ",white_score)
    # print("black:, ", black_score)
    return (white_score - black_score) * 5

def heuristic_eval(board):
    if bool(board.status() & chess.STATUS_NO_BLACK_KING):
        return math.inf
    if bool(board.status() & chess.STATUS_NO_WHITE_KING):
        return -math.inf
    total = material_eval(board) + piece_table_eval(board) + king_attack_eval(board)
    # print("heuristic debug:")
    # print(material_eval(board))
    # print(piece_table_eval(board))
    # print(king_attack_eval(board))
    return total
# calculate actual evaluation on position
def evaluation(board):
    # eval = (heuristic_eval(board) + nn_eval(board)) / 2.0
    return heuristic_eval(board)

# traverse moves that seem to offer a greater heuristic score over others
def determineMoveTraversalOrder(board, possibMoves, isMaximizing):
    scores_to_move_dict = {}
    scores = []
    for idx, move in enumerate(possibMoves):
        board.push(move)
        score = heuristic_eval(board)
        scores.append(score)
        scores_to_move_dict[score] = move
        board.pop()
    if isMaximizing:
        scores.sort(reverse = True) #highest scores at start of list
    else:
        scores.sort() #lowest scores at start of list
    moves = []
    for score in scores:
        moves.append(scores_to_move_dict[score])
    return moves

def shouldPerformSecondSearch(board, isMaximizing, initialMove):
    # if initialMove[0] == -1:
    #     return False
    # last_move = board.peek()
    # # last_move = initialMove
    # initalBoard = initialMove[1]
    # if initalBoard.is_capture(initialMove[0]):
    #     # print("y")
    #     # return True
    #     landed_on = last_move.to_square
    #     if isMaximizing:
    #         if board.is_attacked_by(color = False, square = landed_on):
    #             return True
    #     else:
    #         if board.is_attacked_by(color = True, square = landed_on):
    #             return True
    return False

def minimax(board, depth, isMaximizing, alpha = -math.inf, beta = math.inf, initialMove = (-1, None)):
    if depth == 0:
        if shouldPerformSecondSearch(board, isMaximizing, initialMove):
            eval = minimax(board, 2, isMaximizing, alpha = -math.inf, beta = math.inf, initialMove = (board.peek(), chess.Board(fen = board.board_fen())))[1]
            return (None, eval, None)
        return (None, evaluation(board), None)
    if bool(board.status() & chess.STATUS_NO_BLACK_KING):
        return (None, math.inf, None)
    if bool(board.status() & chess.STATUS_NO_WHITE_KING):
        return (None, -math.inf, None)

    if isMaximizing:
        bestVal = -math.inf
        bestMove = None

        possibMoves = list(board.legal_moves)
        moveTable = {}

        movesToTraverse = determineMoveTraversalOrder(board, possibMoves, isMaximizing)
        for idx, move in enumerate(movesToTraverse):
            nextInitMove = initialMove
            if initialMove[0] == -1:
                nextInitMove = (move, chess.Board(fen = board.board_fen()))
            board.push(move)
            val = minimax(board, depth - 1, False, alpha, beta, nextInitMove)[1]
            board.pop()

            moveTable[idx] = (move, val)

            if val > bestVal:
                bestVal = val
                bestMove = move
            alpha = max(bestVal, alpha)
            if beta <= alpha:
                break
        return (bestMove, bestVal, moveTable)
    else:
        bestVal = math.inf
        bestMove = None

        possibMoves = list(board.legal_moves)
        moveTable = {}

        movesToTraverse = determineMoveTraversalOrder(board, possibMoves, isMaximizing)
        for idx, move in enumerate(movesToTraverse):
            nextInitMove = initialMove
            if initialMove == -1:
                nextInitMove = move
            board.push(move)
            val = minimax(board, depth - 1, True, alpha, beta, nextInitMove)[1]
            board.pop()
            
            moveTable[idx] = (move, val)

            if val < bestVal:
                bestVal = val
                bestMove = move
            beta = min(bestVal, beta)
            if beta <= alpha:
                break;
        return (bestMove, bestVal, moveTable)
pos = chess.Board(fen = "rn2kb1r/pp1b2pp/5n2/q3ppB1/NP1p4/2PP1N2/P4PPP/2RQKB1R b Kkq - 0 10")
#print(pos)
#print(nn_eval(pos))
#print(heuristic_eval(pos))
#print(evaluation(pos))

# move = pos.peek()
# print(pos.is_capture(move))

move, val, table = minimax(board = pos, depth = 3, isMaximizing = False)
#print("move: ", move)
#print("table: ", table)
#print("val: ", val)


# NOTE ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~
# To run engine, call minimax, 
# input the chess.Board position, 
# input the depth of search (more than 3 takes more than a couple of seconds),
# input the isMaximizing, which is the color player, isMaximizing for best possible moves for White, !isMaximizing for best moves for Black
# more negative scores = better for black
# more positive scores = better for white
# ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~


