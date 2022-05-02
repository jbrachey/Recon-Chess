# Recon-Chess
CS4649 Recon Blind Multi-Chess course project

jbrachey3_jdill33_hghori6_chess_engine.py:

This file handles everything pertaining to determining the best move to make given a board decision. The minimax() function, specifically, returns the "best" move with it's associated color for the specified player, along with the scores of all possible moves to be made in the given board position.
It makes the evaluation based on some hand made heuristics. Originally, a neural network was also used for consideration, but this was found to be ineffective compared to hand made heuristic evaluation.

jbrachey3_jdill33_hghori6_chessCNN.py:

This file was used as an attempt to build a neural network to evaluate a given chess position. jbrachey3_jdill33_hghori6_ChessPositionsDataset.py and jbrachey3_jdill33_hghori6_make_data.py were the files used to parse chess board information along with evaluations into something parsable for the neural network. The jbrachey3_jdill33_hghori6_runner.py file was used to train the neural network given the parsed data. The neural network was not ultimately used in the end, however. This was because there was overfitting in the model that I failed to overcome given time constraints. It's still included in the submission because I want to show I accomplished something all the same.

jbrachey3_jdill33_hghori6_mcts.py:

This was was used as an attempt to evaluate the best move given a board state in the same vein as chess_engine.py is intended with minimax search. The mcts implementation used minimax search for its rollout. However, results were meager in comparison to relying on minimax search solely, so this was not used in the final engine as well.

jbrachey3_jdill33_hghori6_ParticleFilter.py:

This was used for handling the uncertainty of the possible chess board states. The particle filter generates new particles every turn and parses out the most likely state in order for the engine to be able to determine a move.