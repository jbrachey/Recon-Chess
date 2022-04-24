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
  
  def best_UCTS(self, extra=0.0001, c=2):
    #use extra so we don't have division by 0

    child_ucts = [(child.score/ (child.num_sims + extra)) + c * math.sqrt(math.log(self.total_visits())/ child.num_visits) for child in self.children]
    return self.children[np.argmax(child_ucts)]
  
  def backpropagate(self, result):
    if self != None:
      self.num_visits += 1
      self.score += result
      node = self.parent
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
    return self.score/ self.num_visits + 2 * math.sqrt(math.log(total_visits)/ self.num_visits)
  
  #add nodes for all possible actions thing could take
  def node_expansion(self):
    for move in self.state.legal_moves:
      new_state = self.state.copy()
      new_state.push(move)
      node = MCTSNode(state=new_state, black=not self.black, agent_turn= not self.agent_turn, 
      parent = self) #can I do this with self? I need this to point back to our current node
    

  #need some bot to determine the moves they will make for this
  #for now we will just use a random move
  def rollout(self):
    new_board = self.state.copy()
    count = 0
    while count < self.DEPTH_OF_SEARCH or self.is_terminal(new_board):
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



  def opponent_move(self, board):
    #fill with agent for predicting this
    return random.choice(list(board.legal_moves))
  
  def agent_move(self, board):
    #fill with agent predicting this
    return random.choice(list(board.legal_moves))
  
  def get_visits(self):
    return self.num_visits
  
  def tree_policy(self):
    curr_node = self
    while not curr_node.is_terminal():
      if curr_node.is_expanded():
        curr_node = curr_node.best_UCTS()
      else:
        return curr_node

  def best_action(self):
    for i in range(self.NUM_ITERATIONS):
      node = self.tree_policy()
      reward = node.rollout()
      node.backpropagate(reward)

    return self.best_UCTS(c=0)

  def is_terminal(self, board):
    return board.is_stalemate() or board.is_checkmate()
  
  def is_expanded(self):
    return self.num_visits != 0



    

