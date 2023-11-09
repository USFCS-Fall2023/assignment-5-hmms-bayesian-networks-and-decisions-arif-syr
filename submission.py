import argparse
import HMM
import sys


def run_viterbi():
    with open(args.viterbi, 'r') as f:
        observations = f.readlines()
        for i, line in enumerate(observations):
            if line != "\n":
                if i % 2 == 0:
                    expected.append(line)
                else:
                    lines.append(line)

    print("Testing the viterbi algorithm:")
    for i, observation in enumerate(lines):
        if len(expected) == len(lines):
            print("Expected:", expected[i], end='')
        best_path = hmm.viterbi(observation)
        print("Actual:", ' '.join(best_path))
        print(observation)

def run_forward():
    with open(args.forward, 'r') as f:
        observations = f.readlines()
        for i, line in enumerate(observations):
            if line != "\n":
                if i % 2 == 0:
                    expected.append(line)
                else:
                    lines.append(line)

    print("Testing the forward algorithm:")
    for i, observation in enumerate(lines):
        if len(expected) == len(lines):
            print("Expected:", expected[i], end='')
        probability_matrix = hmm.forward_algorithm(observation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates an HMM")
    parser.add_argument("--script", help="Script to run, eg: hmm.py")
    parser.add_argument("--basename", help="basename for emissions and transmission")
    parser.add_argument("--forward", help="Filename to run forward algorithm on")
    parser.add_argument("--viterbi", help="Filename to run viterbi algorithm on")
    args = parser.parse_args()

    hmm = HMM.HMM()
    hmm.load(args.basename)
    print("Generated text: ")
    print(hmm.generate(20))

    lines = []  # To store observations
    expected = []  # To store answers
    if args.forward:
        run_forward()

    if args.viterbi:
        run_viterbi()
