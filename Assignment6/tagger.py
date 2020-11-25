import numpy as np
from hmm import HMM
from collections import defaultdict

def model_training(train_data, tags):
    """
    Train an HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class 
            defined in data_process.py (read the file to see what attributes this class has)
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - model: an object of HMM class initialized with parameters (pi, A, B, obs_dict, state_dict) calculated 
            based on the training dataset
    """

    # unique_words.keys() contains all unique words
    unique_words = get_unique_words(train_data)
    
    word2idx = {}
    tag2idx = dict()
    S = len(tags)

    index = 0
    for word in unique_words:
        if word not in word2idx.keys():
            word2idx[word] = index
            index = index + 1
    for i in range(S):
        tag2idx[tags[i]] = i

    pi = np.zeros(S)
    A = np.zeros((S, S))
    B = np.zeros((S, len(unique_words)))

    first_tags_count_dict = defaultdict(int)
    tags_count_dict = defaultdict(int)
    total_transitions_dict = defaultdict(int)
    pos_to_word_dict = defaultdict(lambda: defaultdict(int))
    tag_transition_count_dict = defaultdict(lambda: defaultdict(int))

    for sentence in train_data:
        last_tag = ""
        first_tags_count_dict[sentence.tags[0]] = first_tags_count_dict[sentence.tags[0]] + 1
        for i, word in enumerate(sentence.words):
            current_tag = sentence.tags[i]
            tags_count_dict[current_tag] += 1
            total_transitions_dict[last_tag] += 1
            pos_to_word_dict[current_tag][word] += 1
            tag_transition_count_dict[last_tag][current_tag] += 1
            last_tag = current_tag

    for tag in tags:
        tag_index = tag2idx[tag]
        pi[tag_index] = first_tags_count_dict[tag] / len(train_data)
        for word, count in pos_to_word_dict[tag].items():
            B[tag_index][word2idx[word]] = count / tags_count_dict[tag]
        for next_tag in tags:
            A[tag_index][tag2idx[next_tag]] = tag_transition_count_dict[tag][next_tag] / total_transitions_dict[tag]

    # DO NOT MODIFY BELOW
    model = HMM(pi, A, B, word2idx, tag2idx)
    return model


def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of the "line" class
    - model: an object of the HMM class
    - tags: (1*num_tags) a list containing all possible POS tags

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    S = len(tags)
    next_avlbl_index = max(model.obs_dict.values()) + 1
    z = np.full((S, 1), 10 ** -6)
    for sentence in test_data:
        for word in sentence.words:
            if word not in model.obs_dict.keys():
                model.obs_dict[word] = next_avlbl_index
                model.B = np.append(model.B, z, axis=1)
                next_avlbl_index += 1
        tagging.append(model.viterbi(sentence.words))
    return tagging

# DO NOT MODIFY BELOW
def get_unique_words(data):

    unique_words = {}

    for line in data:
        for word in line.words:
            freq = unique_words.get(word, 0)
            freq += 1
            unique_words[word] = freq

    return unique_words
