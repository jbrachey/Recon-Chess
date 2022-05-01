import numpy as np
import math
import chess
import random
from normal_chess_engine_files import chess_engine
#import chess_engine
from collections import defaultdict
import value_of_pieces


# tree traversal
# num_wins/num_sims + c*sqrt(t)/ni
# num_wins = num wins after the i-th moce
# num_sims = num sims after i-th move
# c = exploration parameter (theoretically equiv to sqrt(2))
# t = total num of sims for parent node

class MCTSNode():
    DEPTH_OF_SEARCH = 6
    NUM_ITERATIONS = 20
    BLACK = False
    WHITE = True

    def __init__(self, state, black, agent_turn, move=None, parent=None):
        self.state = state
        self.black = black  # boolean to figure out player color
        self.agent_turn = agent_turn  # defines whose turn it is
        self.parent = parent
        self.children = []
        self.num_visits = 0
        self.score = 0
        self.move = move

    def best_UCTS(self, extra=0.0001, c=2):
        # use extra so we don't have division by 0

        child_ucts = [(child.score / (child.num_visits + extra)) + c * math.sqrt(
            math.log(self.total_visits()) / (child.num_visits + extra)) for child in self.children]
        return self.children[np.argmax(child_ucts)]

    def backpropagate(self, result):
        self.num_visits += 1
        self.score += result
        node = self.parent
        if node != None:
            node.backpropagate(result)

    def total_visits(self):
        if self.parent == None:
            return self.num_visits
        node = self.parent
        while node.parent != None:
            node = node.parent
        return node.num_visits

    def exploration_eq(self):
        total_visits = self.total_visits()
        if self.num_visist == 0:
            return float("inf")
        return self.score / self.num_visits + 2 * math.sqrt(math.log(total_visits) / self.num_visits)

    # add nodes for all possible actions thing could take
    def node_expansion(self):
        for move in self.state.legal_moves:
            new_state = self.state.copy()
            new_state.push(move)
            node = MCTSNode(state=new_state, black=not self.black, agent_turn=not self.agent_turn,
                            move=move,
                            parent=self)  # can I do this with self? I need this to point back to our current node
            self.children.append(node)

    # need some bot to determine the moves they will make for this
    # for now we will just use a random move
    def rollout(self):
        new_board = self.state.copy()
        count = 0
        while count < self.DEPTH_OF_SEARCH and not self.is_terminal(new_board):
            count += 1
            if self.agent_turn:
                new_board.push(self.agent_move(new_board))
            else:
                move = self.opponent_move(new_board)
                try:
                    new_board.push(move)
                except AttributeError:
                    print('MOVE DIDNT WORK IN ROLLOUT: ', move)
                    new_board.turn = not new_board.turn
            self.agent_turn = not self.agent_turn
        if new_board.is_stalemate():
            return 0
        if bool(new_board.status() & chess.STATUS_NO_BLACK_KING):
            return math.inf
        if bool(new_board.status() & chess.STATUS_NO_WHITE_KING):
            return -math.inf
        return self.value_of_board(new_board)

    """
    # value the board
    # start with just measuring value of pieces on both sides of board
    # your value - opponents value
    # CHANGE TO JESSIE'S VERSION
    def value_of_board(self, board):
        pawns = len(board.pieces(chess.PAWN, chess.WHITE)) - len(board.pieces(chess.PAWN, chess.BLACK))
        knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) - len(board.pieces(chess.KNIGHT, chess.BLACK))
        bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) - len(board.pieces(chess.BISHOP, chess.BLACK))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) - len(board.pieces(chess.ROOK, chess.BLACK))
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) - len(board.pieces(chess.QUEEN, chess.BLACK))

        score = pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens

        

        if self.black:
            return -score
        return score
    """

    def value_of_board(self, board):
        #print("val: ", chess_engine.evaluation(board))
        val = chess_engine.evaluation(board)
        if self.black:
            val = -val
        return val

    def opponent_move(self, board):
        if not self.is_terminal(board):
            move, val, table = chess_engine.minimax(board=board, depth=1, isMaximizing= self.black)
            #move = random.choice(list(board.legal_moves))
        return move
    def agent_move(self, board):
        if not self.is_terminal(board):
            move, val, table = chess_engine.minimax(board=board, depth=1, isMaximizing= not self.black)
            #print('table in mcts: ', table)
            #print(table)
            #move = random.choice(list(board.legal_moves))
        return move

    def get_visits(self):
        return self.num_visits

    def tree_policy(self):
        curr_node = self
        while not curr_node.is_terminal(curr_node.state):
            if curr_node.is_expanded():

                curr_node = curr_node.best_UCTS()
            else:
                #if curr_node == None:
                    #print("self: ", self.state)
                    #print(self.is_terminal(self.state))
                return curr_node
        return curr_node

    def best_action(self):
        for i in range(self.NUM_ITERATIONS):
            node = self.tree_policy()
            #if node == None:
                #print("self node: ", self.state)
            reward = node.rollout()
            node.backpropagate(reward)
            node.node_expansion()

        return self.best_UCTS(c=0).move

    def is_terminal(self, board):
        return board.is_stalemate() or board.is_checkmate()

    def is_expanded(self):
        return self.num_visits != 0


class TestMCTS:
    def main(self):
        board = chess.Board()
        black_move = False
        agent_turn = False
        while not board.is_stalemate() and not board.is_checkmate():
            #print("pre-board")
            #print(board)
            mcts = MCTSNode(state=board, black=black_move, agent_turn=agent_turn)
            #might want best action to return the value of the move too
            move = mcts.best_action()
            board.push(move)
            #print("post-board")
            #print(board)
            black_move = not black_move
            agent_turn = not agent_turn


#test = TestMCTS()
#test.main()
