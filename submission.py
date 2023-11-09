import HMM

if __name__ == '__main__':
    hmm = HMM.HMM()
    hmm.load("partofspeech.browntags.trained")
    print(hmm.generate(20))