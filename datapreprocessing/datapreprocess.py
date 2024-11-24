from nltk_utils import tokenize, stem, bag_of_words

import numpy as np
import json



class DataPreProcess:
    def __init__(self, vectorizer, apply_stemming=False):
        self.vectorizer = vectorizer
        self.__apply_stemming = apply_stemming

    def extract_word_tokens_and_labels(self, data):
        all_word_tokens = []
        tags = []
        pattern_tag_pair = []

        for intent in data['intents']:
            # find the tag word for each intent entry
            tag = intent['tag']
            tags.append(tag)


            # extract the tokens from the corresponding pattern
            for pattern in intent['patterns']:
                word_tokens = tokenize(pattern)
                all_word_tokens.extend(word_tokens)
                pattern_tag_pair.append( (word_tokens, tag) ) # a list of word_token and tag
            
        return (all_word_tokens, tags, pattern_tag_pair)

                
    def apply_stemming(self, all_words, ignore_words):
        if self.__apply_stemming:
            all_words = [stem(w) for w in all_words if w not in ignore_words]
                
        return all_words
    
    def apply_vectorization_and_create_training_data(self, word_tokens_and_tag_pairs, all_word_tokens, tags):
        train_X = []
        train_Y = []

        for (word_tokens, tag) in word_tokens_and_tag_pairs:
            vector_tokens = self.vectorizer(word_tokens, all_word_tokens)
            train_X.append(vector_tokens)
            label = tags.index(tag)
            train_Y.append(label)

        return (train_X, train_Y)
    

def prepare_chatbot_training_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    pre_processing_obj = DataPreProcess(bag_of_words, apply_stemming=True)
    all_word_tokens, tags, word_tokens_and_tag_pair = pre_processing_obj.extract_word_tokens_and_labels(data)
    ignore_words = ['?', '.', '!']
    all_word_tokens = pre_processing_obj.apply_stemming(all_word_tokens, ignore_words)
    all_word_tokens = sorted( set(all_word_tokens) )
    tags = sorted( set(tags) )
    print(tags)
    X_train, Y_train = pre_processing_obj.apply_vectorization_and_create_training_data(word_tokens_and_tag_pair, all_word_tokens, tags)
    return ( np.array(X_train), np.array(Y_train), tags, all_word_tokens )




                
