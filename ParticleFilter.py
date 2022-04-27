import chess
import random
import numpy as np


class ParticleFilter:
    def __init__(self):
        self.numParticles = 1000
        self.particles = create_initial_particle_filter(self.numParticles)

    def handle_sense_result(self, sense_result):
        newParticleFilter = self.particles.copy()
        for count in range(len(newParticleFilter)):
            if not board_agrees_with_sense_result(newParticleFilter[count][0], sense_result):
                newParticle = (newParticleFilter[count][0], 0)
                newParticleFilter[count] = newParticle
        weightedParticleFilter = reweight(newParticleFilter)
        self.particles = sample_new_particles(weightedParticleFilter)

    def update_for_piece_captured(self, captured_square):
        particles = []
        additionalParticles = []
        additionalParticlesProbs = []
        for particle in self.particles:
            board = particle[0]
            legal_moves = list(board.legal_moves)
            possible_moves = []
            probs = []
            for move in legal_moves:
                if move.to_square == captured_square:
                    hasPossibleMove = True
                    possible_moves.append(move)
                    piece = board.piece_at(move.from_square)
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
                        # 1 seemed like too much
                        probs.append(3/4)
            if len(possible_moves) == 0:
                part = (particle[0], 0)
                particles.append(part)
            elif len(possible_moves) == 1:
                board.push(possible_moves[0])
                newParticle = (board, particle[1])
                particles.append(newParticle)
            else:
                probsSum = sum(probs)
                newProbs = [x / probsSum for x in probs]
                move = random.choices(possible_moves, weights=newProbs, k=1)
                moveIndex = possible_moves.index(move)
                particleProb = particle[1] * newProbs[moveIndex]
                possible_moves.pop(moveIndex)
                newProbs.pop(moveIndex)
                for count in range(len(possible_moves)):
                    boardCopy = board.copy()
                    boardCopy.push(possible_moves[count])
                    newPartProb = particle[1] * newProbs[count]
                    newPart = (boardCopy, newPartProb)
                    additionalParticles.append(newPart)
                    additionalParticlesProbs.append(newPartProb)
                board.push(move)
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
        self.particles = reweight(particles)

    def update_no_piece_captured(self):
        particles = []
        for particle in self.particles:
            if random.random() > 0.8:
                # Update this board for random move
                board = particle[0]
                # Will this automatically return moves for black since it's black's turn?
                # Or will it return moves for white? Need to check this.
                legal_moves = list(board.legal_moves)
                board.push(random.choice(legal_moves))
                newParticle = (board, particle[1])
                particles.append(newParticle)
            else:
                # Update this board for one of the best moves, randomly sampled
                # For now, will again just randomly sample out of possible moves
                board = particle[0]
                # Will this automatically return moves for black since it's black's turn?
                # Or will it return moves for white? Need to check this.
                legal_moves = list(board.legal_moves)
                board.push(random.choice(legal_moves))
                newParticle = (board, particle[1])
                particles.append(newParticle)
        self.particles = particles

def create_initial_particle_filter(numParticles):
    particles = []
    weight = 1 / numParticles
    board = chess.Board()
    for _ in range(numParticles):
        particle = (board, weight)
        particles.append(particle)
    return particles

def board_agrees_with_sense_result(board, sense_result):
    for square in sense_result:
        pieceOnBoard = board.piece_at(chess.parse_square(square[0]))
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
