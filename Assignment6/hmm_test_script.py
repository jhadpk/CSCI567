from hmm import HMM
import numpy as np
import json
import time
from data_process import Dataset
from util import accuracy
from tagger import model_training, sentence_tagging

def hmm_test():

    st_time = time.time()

    model_file = "hmm_model.json"

    # load data
    with open(model_file, 'r') as f:
        data = json.load(f)
    A = np.array(data['A'])
    B = np.array(data['B'])
    pi = np.array(data['pi'])
    # observation symbols
    obs_dict = data['observations']
    # state symbols
    states_symbols = dict()
    for idx, item in enumerate(data['states']):
        states_symbols[item] = idx
    Osequence = np.array(data['Osequence'])
    N = len(Osequence)
    model = HMM(pi, A, B, obs_dict, states_symbols)

    alpha = model.forward(Osequence)
    m_alpha = np.array([[3.5000e-01, 1.3600e-01, 0.0000e+00, 0.0000e+00, 1.1136e-05, 1.1136e-05, 0.0000e+00],
               [1.5000e-01, 3.2000e-02, 4.6400e-03, 2.7840e-04, 3.3408e-05, 1.1136e-05, 8.9088e-07]])

    print("Your forward function output:\n", alpha)
    print("Correct forward function output:\n", m_alpha)

    beta = model.backward(Osequence)
    m_beta = np.array([[1.6896e-06, 3.8400e-06, 6.4000e-05, 2.0000e-03, 1.4000e-02, 2.0000e-02, 1.0000e+00],
               [1.9968e-06, 1.1520e-05, 1.9200e-04, 3.2000e-03, 2.2000e-02, 6.0000e-02, 1.0000e+00]])

    print("Your backward function output:\n", beta)
    print("Correct backward function output:\n", m_beta)

    prob1 = model.sequence_prob(Osequence)
    m_prob1 = 8.908800000000002e-07

    print("Your sequence_prob function output:\n", prob1)
    print("Correct sequence_prob function output:\n", m_prob1)

    prob2 = model.posterior_prob(Osequence)
    m_prob2 = np.array([[0.6637931, 0.5862069, 0., 0., 0.175, 0.25, 0.],
               [0.3362069, 0.4137931, 1., 1., 0.825, 0.75, 1.]])

    print("Your posterior_prob function output:\n", prob2)
    print("Correct posterior_prob function output:\n", m_prob2)

    prob3 = model.likelihood_prob(Osequence)
    m_prob3 = np.array(
        [[[0.48275862, 0,         0, 0, 0.1,   0.],
          [0.18103448, 0.5862069, 0, 0, 0.075, 0.25]],
         [[0.10344828, 0.,        0, 0.175, 0.15, 0.],
          [0.23275862, 0.4137931, 1, 0.825, 0.675, 0.75]]]
        )
 
    print("Your likelihood_prob function output:\n", prob3)
    print("Correct likelihood_prob function output:\n", m_prob3)

    viterbi_path = model.viterbi(Osequence)
    m_viterbi_path = ['1', '1', '2', '2', '2', '2', '2']

    print('Your viterbi function output: \n', viterbi_path)
    print('Correct viterbi function output: \n', m_viterbi_path)

    en_time = time.time()
    print()
    print("HMM total time: ", en_time - st_time)


def speech_tagging_test():
    st_time = time.time()
    data = Dataset("pos_tags.txt", "pos_sentences.txt", train_test_split=0.8, seed=0)

    data.train_data = data.train_data[:100]

    data.test_data = data.test_data[:10]

    model = model_training(data.train_data, data.tags)

    tagging = sentence_tagging(data.test_data, model, data.tags)

    total_words = 0
    total_correct = 0
    for i in range(len(tagging)):
        correct, words, accur = accuracy(tagging[i], data.test_data[i].tags)
        total_words += words
        total_correct += correct
        #print("accuracy: ", accur)

    print("Your total accuracy for tagging: ", total_correct*1.0/total_words)
    print("Expected total accuracy is about: ", 0.76)

    en_time = time.time()
    print("Sentence tagging total time: ", en_time - st_time)


if __name__ == "__main__":

    hmm_test()

    speech_tagging_test()
