

import random
import numpy as np
import argparse
import codecs
import os
import numpy

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        self.emissions = {}
        self.transitions = {}

        with open(f"{basename}.emit", 'r') as emission_file:
            for line in emission_file:
                parts = line.split()
                # Use setdefault() to create a nested dictionary if it doesn't exist
                self.emissions.setdefault(parts[0], {})[parts[1]] = float(parts[2])

        with open(f"{basename}.trans", 'r') as transmission_file:
            for line in transmission_file:
                parts = line.split()
                self.transitions.setdefault(parts[0], {})[parts[1]] = float(parts[2])

    def generate(self, n):
        current_state = '#'
        states_sequence = []
        emissions_sequence = []

        for i in range(n):
            # Pick a random successor using probability as weights.
            # Then, pick an emission using that state
            next_state = random.choices(list(self.transitions[current_state].keys()), weights=self.transitions[current_state].values())[0]
            next_emission = random.choices(list(self.emissions[next_state].keys()), weights=self.emissions[next_state].values())[0]

            # To see the highest probability option, and if it was actually chosen (not for testing functionality)
            # max_prob = max(emissions_tmp.values())  # maximum value
            # max_key = [k for k, v in emissions_tmp.items() if v == max_prob]
            # print(max_key, max_prob)
            # print(next_emission)
            # print()

            current_state = next_state
            states_sequence.append(next_state)
            emissions_sequence.append(next_emission)
        return Observation(states_sequence, emissions_sequence)

    # Converts a string of space-separated tokens to an Observation with no state sequence
    def text_to_observation(self, text):
        outputseq = text.strip().split()
        stateseq = [None] * len(outputseq)
        return Observation(stateseq, outputseq)

    def forward_algorithm(self, text, print_predicted_states = True):
        # Convert the text to an Observation
        observation = self.text_to_observation(text)
        states = [state for state in self.transitions.keys() if state != '#']
        num_observations = len(observation.outputseq)
        M = np.zeros((num_observations + 1, len(states)))
        initial_state = '#'

        for s in states:
            if s in self.emissions and observation.outputseq[0] in self.emissions[s]:
                M[1, states.index(s)] = self.transitions[initial_state].get(s, 0) * self.emissions[s].get(
                    observation.outputseq[0], 0)

        for i in range(2, num_observations + 1):
            for s in states:
                sum_prob = 0
                for s2 in states:
                    if observation.outputseq[i - 1] in self.emissions[s]:
                        sum_prob += M[i - 1, states.index(s2)] * self.transitions[s2].get(s, 0) * self.emissions[s].get(
                            observation.outputseq[i - 1], 0)
                M[i, states.index(s)] = sum_prob

        if print_predicted_states:
            state_indices = np.argmax(M[1:], axis=1)
            predicted_states = [states[index] for index in state_indices]
            print("Predicted:", ' '.join(predicted_states))
            print(text)

        return M

    # you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.
    def viterbi(self, text, print_predicted_states = True):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        # Convert the text to an Observation
        observation = self.text_to_observation(text)
        num_observations = len(observation.outputseq)

        states = [state for state in self.transitions.keys() if state != '#']
        M = np.zeros((num_observations + 1, len(states)))
        backpointers = np.zeros((num_observations + 1, len(states)), dtype=int)

        # Ignore '#' while initializing starting probabilities
        for s in states:
            M[1, states.index(s)] = self.transitions['#'].get(s, 0) * self.emissions[s].get(observation.outputseq[0], 0)
            backpointers[1, states.index(s)] = 0

        # Fill in the M and backpointers
        for t in range(2, num_observations + 1):
            for s in states:
                probs = [M[t - 1, states.index(s2)] * self.transitions[s2].get(s, 0) * self.emissions[s].get(
                    observation.outputseq[t - 1], 0) for s2 in states]
                M[t, states.index(s)] = max(probs)
                backpointers[t, states.index(s)] = np.argmax(probs)

        # Backtrack to find the best path
        best_path = []
        best_state = np.argmax(M[num_observations])

        for t in range(num_observations, 0, -1):
            best_path.insert(0, states[best_state])
            best_state = backpointers[t, best_state]

        return best_path





