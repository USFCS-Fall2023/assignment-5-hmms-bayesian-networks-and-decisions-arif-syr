import argparse
import alarm, carnet, HMM
import sys


def run_viterbi():
    lines = []  # To store observations
    expected = []  # To store answers
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
    lines = []  # To store observations
    expected = []  # To store answers
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

    if args.forward:
        run_forward()

    if args.viterbi:
        run_viterbi()

    print("####################################")

    # Storing queries and evidence for alarm.py in here for readability
    alarm_variables = [["MaryCalls"],
                       ["JohnCalls", "MaryCalls"],
                       ["Alarm"]]
    alarm_evidence = [{"JohnCalls": "yes"},
                      {"Alarm": "yes"},
                      {"MaryCalls": "yes"}]
    print("Running alarm.py queries")

    for variable, evidence in zip(alarm_variables, alarm_evidence):
        print(f"Probability of {', '.join(var for var in variable)} given {', '.join(e for e in evidence)}:")
        alarm.run(variable, evidence)

    print("####################################")
    print("Running carnet.py queries")
    carnet.run()