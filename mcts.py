import numpy as np
import math
import chess
import random
from collections import defaultdict
import value_of_pieces

#tree traversal
#num_wins/num_sims + c*sqrt(t)/ni
#num_wins = num wins after the i-th moce
#num_sims = num sims after i-th move
#c = exploration parameter (theoretically equiv to sqrt(2))
#t = total num of sims for parent node

class MCTSNode():
  DEPTH_OF_SEARCH = 6
  NUM_ITERATIONS = 10
  def __init__(self, state, black, agent_turn, parent=None):
    self.state = state
    self.black = black #boolean to figure out player color
    self.agent_turn = agent_turn #defines whose turn it is
    self.parent = parent
    self.children = []
    self.num_visits = 0
    self.score = 0
    self.actions = self.possible_actions()
  
  def total_visits():
    if self.parent == None:
      return self.num_visits
    node = self.parent
    while node.parent != None:
      node = node.parent
    return node.num_visits

  def exploration_eq():
    total_visits = self.total_visits()
    if self.num_visist == 0:
      return float("inf")
    return self.score/ self.num_visits + 2 * math.sqrt(math.log(total_visits)/ self.num_visits)
  
  #add nodes for all possible actions thing could take
  def node_expansion():
    for move in self.state.legal_moves:
      new_state = self.state.copy()
      new_state.push(move)
      node = MCTSNode(state=new_state, black=not self.black, agent_turn= not self.agent_turn, 
      parent = self) #can I do this with self? I need this to point back to our current node
    

  #need some bot to determine the moves they will make for this
  #for now we will just use a random move
  def rollout():
    new_board = self.state.copy()
    count = 0
    while count < self.DEPTH_OF_SEARCH or not new_board.is_stalemate() or not new_board.is_checkmate():
      count += 1
      if self.agent_turn:
        new_board.push(self.agent_move(new_board))
      else:
        new_board.push(self.opponent_move(new_board))
      self.agent_turn = not self.agent_turn 
    if new_board.is_stalemate():
      return 0
    if new_board.is_checkmate():
      if self.agent_turn:
        #loss
        return -100
      else:
        return 100
    return self.value_of_board(new_board)

  #value the board
  #start with just measuring value of pieces on both sides of board
  #your value - opponents value
  def value_of_board(board):
    pawns = len(board.pieces(chess.PAWN, chess.WHITE)) - len(board.pieces(chess.PAWN, chess.BLACK))
    knights = len(board.pieces(chess.KNIGHT, chess.WHITE)) - len(board.pieces(chess.KNIGHT, chess.BLACK))
    bishops = len(board.pieces(chess.BISHOP, chess.WHITE)) - len(board.pieces(chess.BISHOP, chess.BLACK))
    rooks = len(board.pieces(chess.ROOK, chess.WHITE)) - len(board.pieces(chess.ROOK, chess.BLACK))
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) - len(board.pieces(chess.QUEEN, chess.BLACK))

    score = pawns + 3 * knights + 3 * bishops + 5 * rooks + 9 * queens
    if self.black:
      return -score
    return score



  def opponent_move(board):
    pass
  
  def agent_move(board):
    pass
  
  def get_visits():
    return self.num_visits

  def main():
    self.expansion()
    

    

