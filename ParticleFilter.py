import chess
import random


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