

import random
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

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        self.emissions = {}
        self.transitions = {}
        with open(f"{basename}.emit", 'r') as emission_file:
            for line in emission_file:
                parts = line.split()
                self.emissions.setdefault(parts[0], {})[parts[1]] = float(parts[2])

        with open(f"{basename}.trans", 'r') as transmission_file:
            for line in transmission_file:
                parts = line.split()
                self.transitions.setdefault(parts[0], {})[parts[1]] = float(parts[2])




   ## you do this.
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



    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """





