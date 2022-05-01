#!/usr/bin/env python3

"""
File Name:      my_agent.py
Authors:        TODO: Your names here!
Date:           TODO: The date you finally started working on this.

Description:    Python file for my agent.
Source:         Adapted from recon-chess (https://pypi.org/project/reconchess/)
"""

import random
import chess
from player import Player
from ParticleFilter import ParticleFilter
import mcts


# TODO: Rename this class to what you would like your bot to be named during the game.
class MyAgent(Player):

    def __init__(self):
        self.numParticles = 1000
        self.particle_filter = ParticleFilter()
        self.board = None
        self.white = None

    def handle_game_start(self, color, board):
        """
        This function is called at the start of the game.

        :param color: chess.BLACK or chess.WHITE -- your color assignment for the game
        :param board: chess.Board -- initial board state
        :return:
        """
        # TODO: implement this method
        self.white = color
        self.board = board

    def handle_opponent_move_result(self, captured_piece, captured_square):
        """
        This function is called at the start of your turn and gives you the chance to update your board.

        :param captured_piece: bool - true if your opponents captured your piece with their last move
        :param captured_square: chess.Square - position where your piece was captured
        """
        if captured_piece:
            self.particle_filter.update_for_piece_captured(captured_square)
        else:
            self.particle_filter.update_no_piece_captured()

    def choose_sense(self, possible_sense, possible_moves, seconds_left):
        """
        This function is called to choose a square to perform a sense on.

        :param possible_sense: List(chess.SQUARES) -- list of squares to sense around
        :param possible_moves: List(chess.Moves) -- list of acceptable moves based on current board
        :param seconds_left: float -- seconds left in the game

        :return: chess.SQUARE -- the center of 3x3 section of the board you want to sense
        :example: choice = chess.A1
        """
        # TODO: update this method
        #ideas we could minimize entropy. Take our possible board states and
        #we can use possible_moves to decrease our possible boards
        board_states = self.particle_filter.get_most_probable_board_states()
        probability = 0
        differences = {}
        max_square = None
        for square in possible_sense:
            differences[square] = set()
            for board in board_states:
                #arbitrary constant to say we have sampled enough boards
                #goal is to minimize entropy aka eliminate as many board possibilities as possible
                if probability > 500:
                    break
                probability += board[1]
                if board[0].piece_at(square) not in differences[square]:
                    differences[square].add(board[0].piece_at(square))
        best_square = (None, -1)
        for key in differences.keys():
            if best_square[1] < len(differences[key]):
                best_square = (key, len(differences[key]))

        return best_square[0]

    def handle_sense_result(self, sense_result):
        """
        This is a function called after your picked your 3x3 square to sense and gives you the chance to update your
        board.

        :param sense_result: A list of tuples, where each tuple contains a :class:`Square` in the sense, and if there
                             was a piece on the square, then the corresponding :class:`chess.Piece`, otherwise `None`.
        :example:
        [
            (A8, Piece(ROOK, BLACK)), (B8, Piece(KNIGHT, BLACK)), (C8, Piece(BISHOP, BLACK)),
            (A7, Piece(PAWN, BLACK)), (B7, Piece(PAWN, BLACK)), (C7, Piece(PAWN, BLACK)),
            (A6, None), (B6, None), (C8, None)
        ]
        """
        # TODO: implement this method
        # Hint: until this method is implemented, any senses you make will be lost.
        self.particle_filter.handle_sense_result(sense_result)

    def choose_move(self, possible_moves, seconds_left):
        """
        Choose a move to enact from a list of possible moves.

        :param possible_moves: List(chess.Moves) -- list of acceptable moves based only on pieces
        :param seconds_left: float -- seconds left to make a move

        :return: chess.Move -- object that includes the square you're moving from to the square you're moving to
        :example: choice = chess.Move(chess.F2, chess.F4)

        :condition: If you intend to move a pawn for promotion other than Queen, please specify the promotion parameter
        :example: choice = chess.Move(chess.G7, chess.G8, promotion=chess.KNIGHT) *default is Queen
        """
        # TODO: update this method
        #update this so we are sampling more states and taking the weighted best move
        guess_state = self.sample_states()
        monte_carlo_tree_search = mcts.MCTSNode(state=guess_state, black= not guess_state.turn, agent_turn=guess_state.turn == self.white)

        move = monte_carlo_tree_search.best_action()
        return move

    def sample_states(self):
        states = self.particle_filter.get_most_probable_board_states()
        boards = []
        weights = []
        for state in states:
            boards.append(state[0])
            weights.append(state[1])
        return random.choice(boards, weights=weights, k=1)

    def handle_move_result(self, requested_move, taken_move, reason, captured_piece, captured_square):
        """
        This is a function called at the end of your turn/after your move was made and gives you the chance to update
        your board.

        :param requested_move: chess.Move -- the move you intended to make
        :param taken_move: chess.Move -- the move that was actually made
        :param reason: String -- description of the result from trying to make requested_move
        :param captured_piece: bool - true if you captured your opponents piece
        :param captured_square: chess.Square - position where you captured the piece
        """
        # TODO: implement this method
        if requested_move == taken_move:
            self.particle_filter.update_for_requested_move(taken_move, captured_piece)
        else:
            self.particle_filter.update_for_unrequested_move(requested_move, taken_move, captured_piece, captured_square)

    def handle_game_end(self, winner_color, win_reason):  # possible GameHistory object...
        """
        This function is called at the end of the game to declare a winner.

        :param winner_color: Chess.BLACK/chess.WHITE -- the winning color
        :param win_reason: String -- the reason for the game ending
        """
        # TODO: implement this method


        pass

    def create_initial_particle_filter(self, numParticles, board):
        particles = []
        weight = 1 / numParticles
        for _ in range(numParticles):
            particle = (board, weight)
            particles.append(particle)
        return particles

    def reweight(self, particleFilter):
        newParticleFilter = []
        totalWeight = 0
        for particle in particleFilter:
            totalWeight += particle[1]
        for particle in particleFilter:
            newParticle = (particle[0], particle[1] / totalWeight)
            newParticleFilter.append(newParticle)
        return newParticleFilter

    def sample_new_particles(self, particleFilter):
        newParticleFilter = []
        particles = []
        probabilities = []
        for particle in particleFilter:
            particles.append(particle[0])
            probabilities.append(particle[1])
        newParticles = random.choices(particles, weights=probabilities, k=self.numParticles)
        for particle in newParticles:
            weightedParticle = (particle, 1 / self.numParticles)
            newParticleFilter.append(weightedParticle)
        return newParticleFilter

    def board_agrees_with_sense_result(self, board, sense_result):
        for square in sense_result:
            pieceOnBoard = board.piece_at(chess.parse_square(square[0]))
            if square[1] is None and pieceOnBoard is None:
                continue
            if (square[1] is None and pieceOnBoard is not None) or (square[1] is not None and pieceOnBoard is None):
                return False
            if pieceOnBoard.color != square[1].color or pieceOnBoard.piece_type != square[1].piece_type:
                return False
        return True

