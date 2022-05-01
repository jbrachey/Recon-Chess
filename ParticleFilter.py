import chess
import random
import numpy as np
from normal_chess_engine_files import chess_engine
#import chess_engine


class ParticleFilter:
    def __init__(self):
        self.numParticles = 2000
        self.particles = create_initial_particle_filter(self.numParticles)
        self.bestSenseSquare = 24

    def handle_sense_result(self, sense_result):
        particles = []
        for particle in self.particles:
            if not board_agrees_with_sense_result(particle[0], sense_result):
                newParticle = (particle[0], particle[1] * 0.0001)
                particles.append(newParticle)
            else:
                newParticle = particle
                particles.append(newParticle)
        weightedParticleFilter = reweight(particles)
        self.particles = sample_new_particles(weightedParticleFilter, self.numParticles)

    def update_for_piece_captured(self, captured_square):
        #print('in update for piece captured')
        particles = []
        to_squares = [0 for _ in range(64)]
        additionalParticles = []
        additionalParticlesProbs = []
        for particle in self.particles:
            board = particle[0].copy()
            legal_moves = list(board.legal_moves)
            possible_moves = []
            probs = []
            for move in legal_moves:
                if move.to_square == captured_square:
                    possible_moves.append(move)
                    piece = board.piece_at(move.from_square).symbol()
                    # This section goes under assumption that opposing player is more likely to capture
                    # with a smaller value piece than with a larger value piece if they can capture with multiple pieces
                    if piece == 'q' or piece == 'Q':
                        probs.append(1/9)
                    elif piece == 'k' or piece == 'K':
                        # Kinda arbitrary, definitely could change
                        probs.append(1/6)
                    elif piece == 'r' or piece == 'R':
                        probs.append(1/5)
                    elif piece == 'n' or piece == 'N' or piece == 'b' or piece == 'B':
                        probs.append(1/3)
                    elif piece == 'p' or piece == 'P':
                        probs.append(3/4)
                    else:
                        probs.append(1/2)

            if len(possible_moves) == 0:
                board.remove_piece_at(captured_square)
                board.turn = not board.turn
                part = (board, particle[1] * 0.001)
                particles.append(part)
            elif len(possible_moves) == 1:
                #print('push 1')
                board.push(possible_moves[0])
                to_squares[possible_moves[0].to_square] += 1
                newParticle = (board, particle[1])
                particles.append(newParticle)
            else:
                probsSum = sum(probs)
                newProbs = [x / probsSum for x in probs]
                #print('num moves: ', len(possible_moves))
                #print('num probs: ', len(newProbs))
                #print('possible moves: ', possible_moves)
                move = random.choices(possible_moves, weights=newProbs, k=1)[0]
                moveIndex = possible_moves.index(move)
                particleProb = particle[1] * newProbs[moveIndex]
                possible_moves.pop(moveIndex)
                newProbs.pop(moveIndex)
                for count in range(len(possible_moves)):
                    boardCopy = board.copy()
                    #print('push 2')
                    boardCopy.push(possible_moves[count])
                    to_squares[possible_moves[count].to_square] += 1
                    newPartProb = particle[1] * newProbs[count]
                    newPart = (boardCopy, newPartProb)
                    additionalParticles.append(newPart)
                    additionalParticlesProbs.append(newPartProb)
                #print('push 3')
                board.push(move)
                to_squares[move.to_square] += 1
                newParticle = (board, particleProb)
                particles.append(newParticle)
        # Now, fill in rest of open particle spaces with additionalParticles entries
        additionalParticlesProbs = np.array(additionalParticlesProbs)
        additionalParticlesProbs /= additionalParticlesProbs.sum()
        if len(particles) < self.numParticles and len(additionalParticles) > 0:
            particlesLeft = self.numParticles - len(particles)
            k = min(particlesLeft, len(additionalParticles))
            sample = np.random.choice(additionalParticles, size=k, replace=False, p=additionalParticlesProbs)
            for p in sample:
                particles.append(p)
        while len(particles) < self.numParticles:
            board = chess.Board()
            part = (board, 0)
            particles.append(part)
        self.bestSenseSquare = get_best_sense_square(to_squares)
        self.particles = sample_new_particles(reweight(particles), self.numParticles)

    def update_no_piece_captured(self):
        #print('in update no piece captured')
        particles = []
        to_squares = [0 for _ in range(64)]
        for particle in self.particles:
            board = particle[0].copy()
            initial_legal_moves = list(board.legal_moves)
            legal_moves = []
            for move in initial_legal_moves:
                if not board.is_capture(move):
                    legal_moves.append(move)
            legal_moves.append('None')
            #if len(legal_moves) < 3:
                #print('BOARD TURN: ', board.turn)
                #print('not many legal moves: ', legal_moves)
                #print('weird board: ', board)
            if random.random() > 0.01:
                # Will this automatically return moves for black since it's black's turn?
                # Or will it return moves for white? Need to check this.
                #print('push 4')
                move = random.choice(legal_moves)
                if move == 'None':
                    board.turn = not board.turn
                else:
                    to_squares[move.to_square] += 1
                    board.push(move)
            else:
                # Update this board for one of the best moves, randomly sampled
                # For now, will again just randomly sample out of possible moves
                #white = particle[2]
                # Will this automatically return moves for black since it's black's turn?
                # Or will it return moves for white? Need to check this.
                #_, table_of_moves, __ = chess_engine.minimax(board=board, depth=3, isMaximizing=white)
                #legal_moves, weights = self.conv_map_of_moves_to_list(table_of_moves)
                #print('board turn: ', board.turn)
                move, val, table = chess_engine.minimax(board=board.copy(), depth=2, isMaximizing=board.turn)
                if table is None:
                    move = random.choice(legal_moves)
                    if move == 'None':
                        board.turn = not board.turn
                    else:
                        to_squares[move.to_square] += 1
                        board.push(move)
                    newParticle = (board, particle[1])
                    particles.append(newParticle)
                    continue
                #print('move: ', move)
                #print("move: ", move)
                #print('move: ', move)
                #print("table: ", table)
                moves = []
                for count in range(len(table)):
                    if count >= 10:
                        break
                    moves.append(table[count])
                if not board.turn:
                    moves.sort(key=lambda k: k[1], reverse=False)
                    weights = [1 / moves[x][1] for x in range(len(moves))]
                else:
                    moves.sort(key=lambda k: k[1], reverse=True)
                    weights = [moves[x][1] for x in range(len(moves))]
                weights = distribute(weights)
                #print('moves: ', moves)
                inputMoves = [moves[x][0] for x in range(len(moves))]
                #print('len(inputMoves): ', len(inputMoves))
                #print('len(weights): ', len(weights))
                if len(inputMoves) == 0 or len(weights) == 0 or len(inputMoves) != len(weights):
                    move = random.choice(legal_moves)
                    if move == 'None':
                        board.turn = not board.turn
                    else:
                        to_squares[move.to_square] += 1
                        board.push(move)
                else:
                    move = random.choices(inputMoves, weights=weights, k=1)[0]
                    if move == 'None':
                        board.turn = not board.turn
                    else:
                        to_squares[move.to_square] += 1
                        board.push(move)
                #board.push(random.choice(legal_moves, weights=weights, k=1))
            newParticle = (board, particle[1])
            particles.append(newParticle)
        self.bestSenseSquare = get_best_sense_square(to_squares)
        self.particles = particles


    def conv_map_of_moves_to_list(self, table_of_moves):
        moves = []
        weights = []
        total_weight = 0
        min_weight = 0
        for key in table_of_moves.keys():
            move = table_of_moves[key][0]
            weight = table_of_moves[key][1]
            moves.append(move)
            weights.append(weight)
            #shiting all the weights to start at 0 so we can easily scale them for weighting the diff moves
            if weight < min_weight:
                total_weight += (min_weight - weight) * len(weights)
            else:
                total_weight += weight - min_weight #min weight will be negative so this adds
        for i, weight in enumerate(weights):
            weights[i] = weight/ total_weight
        return moves, weights

    def update_for_requested_move(self, taken_move, captured_piece):
        #print('in update for requested move')
        particles = []
        for particle in self.particles:
            board = particle[0].copy()
            legal_moves = list(board.legal_moves)
            prob = particle[1]
            # Basically, if the move was valid for this particle, make the move and add the particle with probability
            if taken_move in legal_moves and (
                    (captured_piece and board.piece_at(taken_move.to_square) is not None) or (
                    not captured_piece and board.piece_at(taken_move.to_square) is None)):
                #print('push 6')
                board.push(taken_move)
                prob *= 10
            # If the move was valid but not perfect for board, make move and give it probability 1 / prev probability
            elif taken_move in legal_moves:
                #print('push 7')
                board.push(taken_move)
                prob /= 10
            # If the move was invalid, give this particle probability 0 so it won't be sampled
            else:
                #print('push 8')
                prob *= 0.0001
            newParticle = (board, prob)
            particles.append(newParticle)
        self.particles = sample_new_particles(reweight(particles), self.numParticles)

    def update_for_unrequested_move(self, requested_move, taken_move, captured_piece, captured_square):
        #print('in update for unrequested move')
        particles = []
        for particle in self.particles:
            board = particle[0].copy()
            legal_moves = list(board.legal_moves)
            if taken_move is None:
                #print('taken move is none!')
                board.turn = not board.turn
                probability = particle[1]
                if requested_move in legal_moves:
                    probability *= 0.001
                newParticle = (board, probability)
                particles.append(newParticle)
            # Follow similar logic to update_for_requested_move. If board corresponds perfectly with move outcome,
            # then make the move and add this particle. If the taken_move was possible but the board does not
            # correspond perfectly, then make the move and add this particle with probability prevProb / 10.
            # Finally, if taken_move isn't legal in the board, give a prob of 0 so this particle won't be sampled.
            else:
                if requested_move not in legal_moves and taken_move in legal_moves and \
                        ((captured_piece and board.piece_at(captured_square) is not None) or
                         (not captured_piece and board.piece_at(taken_move.to_square) is None)):
                    #print('push 9')
                    #print('in initial if')
                    board.push(taken_move)
                    newParticle = (board, particle[1] * 10)
                    particles.append(newParticle)
                elif taken_move in legal_moves:
                    #print('push 10')
                    #print('taken move in legal moves! taken move: ', taken_move)
                    board.push(taken_move)
                    newParticle = (board, particle[1] / 10)
                    particles.append(newParticle)
                else:
                    #print('push 11')
                    #print("we're in the else")
                    board.push(list(board.legal_moves)[0])
                    newParticle = (board, particle[1] * 0.0001)
                    particles.append(newParticle)
        self.particles = sample_new_particles(reweight(particles), self.numParticles)

    def get_most_probable_board_states(self):
        self.particles.sort(key=lambda k: k[1], reverse=True)
        # print(self.particles)
        return self.particles

    def get_best_square(self):
        return self.bestSenseSquare

def create_initial_particle_filter(numParticles):
    particles = []
    weight = 1 / numParticles
    board = chess.Board()
    #print('initial board turn: ', board.turn)
    for _ in range(numParticles):
        particle = (board.copy(), weight)
        particles.append(particle)
    #print("create init: ", particles)
    return particles

def board_agrees_with_sense_result(board, sense_result):
    #print(sense_result)
    for square in sense_result:
        pieceOnBoard = board.piece_at(square[0])
        if square[1] is None and pieceOnBoard is None:
            continue
        if (square[1] is None and pieceOnBoard is not None) or (square[1] is not None and pieceOnBoard is None):
            return False
        if pieceOnBoard.color != square[1].color or pieceOnBoard.piece_type != square[1].piece_type:
            return False
    return True

def reweight(particleFilter):
    newParticleFilter = []
    totalWeight = 0
    for particle in particleFilter:
        totalWeight += particle[1]
    for particle in particleFilter:
        newParticle = (particle[0], particle[1] / totalWeight)
        newParticleFilter.append(newParticle)
    return newParticleFilter

def sample_new_particles(particleFilter, numParticles):
    newParticleFilter = []
    particles = []
    probabilities = []
    for particle in particleFilter:
        particles.append(particle[0])
        probabilities.append(particle[1])
    newParticles = random.choices(particles, weights=probabilities, k=numParticles)
    for particle in newParticles:
        weightedParticle = (particle, 1 / numParticles)
        newParticleFilter.append(weightedParticle)
    return newParticleFilter

def get_best_sense_square(to_squares):
    squares = [0 for _ in range(64)]
    max = 0
    maxSquare = 27
    for square in range(9, 55):
        squares[square] += to_squares[square]
        squares[square] += to_squares[square - 1]
        squares[square] += to_squares[square + 1]
        squares[square] += to_squares[square + 8]
        squares[square] += to_squares[square - 8]
        squares[square] += to_squares[square - 9]
        squares[square] += to_squares[square - 7]
        squares[square] += to_squares[square + 7]
        squares[square] += to_squares[square + 9]
        if squares[square] > max:
            max = squares[square]
            maxSquare = square
    return maxSquare

def distribute(weights):
    totalWeight = 0
    newWeights = []
    for weight in weights:
        totalWeight += weight
    if totalWeight == 0:
        return [1 / len(weights) for _ in range(len(weights))]
    for weight in weights:
        newWeights.append(weight / totalWeight)
    return newWeights