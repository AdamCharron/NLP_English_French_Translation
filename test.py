from preprocess import *
from lm_train import *
from log_prob import *
from perplexity import *
from align_ibm1 import *
from evalAlign import *
from BLEU_score import *
import pickle
import os

test_parts = [1,2,3,4,5]
test_parts = [5]

e_model = None
f_model = None
fn_LM = "/h/u16/c8/00/charronh/Desktop/CSC401/A2/"
fn_AM = "/h/u16/c8/00/charronh/Desktop/CSC401/A2/"
data_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
test_dir = "/u/cs401/A2_SMT/data/Hansard/Testing/"
#data_dir = "/h/u16/c8/00/charronh/Desktop/CSC401/A2/"

if 1 in test_parts:
    print("Testing part 1")
    
    repo = "/u/cs401/A2_SMT/data/Hansard/"
    training_subdir = "Training/"
    testing_subdir = "Testing/"

    filename = repo + training_subdir + "hansard.36.1.house.debates.122"
    f_e = open(filename + '.e')
    f_f = open(filename + '.f')
    start_line = 1
    length = 1180
    for i in range(1,start_line + length):
        if i < start_line: 
            f_e.readline()
            f_f.readline()
            continue
        print("\n\n============= Line: {} ================".format(i))
        in_sentence_e = f_e.readline().strip()
        in_sentence_f = f_f.readline().strip()
        
        print("\nIn:")
        print(in_sentence_e)
        print(in_sentence_f)
        
        print("\nOut:")
        print(preprocess(in_sentence_e, 'e'))
        print(preprocess(in_sentence_f, 'f'))
        
if 2 in test_parts:
    print("Testing part 2")
    language = 'e'
    e_model = lm_train(data_dir, language, fn_LM)
    language = 'f'
    f_model = lm_train(data_dir, language, fn_LM)
    
if 3 in test_parts:
    print("Testing part 3")
    smoothing=True 
    delta=0.5
    deltas = [0.1, 0.3, 0.5, 0.7, 0.9]
    vocabSize = 0
    languages = ['e','f']
    for language in languages:
        LM = pickle.load(open(language + '.pickle', 'rb'))
        vocabSize = len(LM['uni'])
        for filename in os.listdir(data_dir):
            if filename.endswith('.' + language):
                f = open(os.path.join(data_dir, filename), 'r')
                for line in f:
                    line = line.rstrip()
                    sentence = preprocess(line, language)
                    log_p = log_prob(sentence, LM, smoothing, delta, vocabSize)
                    #print("{}\tProb: {}".format(sentence, log_p))
        for d in deltas:        
            print("Language: {}, delta: {}, perp: {}".format(language, d, preplexity(LM, test_dir, language, smoothing, d)))
    
if 4 in test_parts:
    print("Testing part 4")
    num_sentences = 1000
    max_iter = 10
    AM = align_ibm1(data_dir, num_sentences, max_iter, fn_AM)
    from pprint import pprint
    pprint(AM)
    
if 5 in test_parts:
    print("Testing part 5")
    fn_LM = "/h/u16/c8/00/charronh/Desktop/CSC401/A2/"
    fn_AM = "/h/u16/c8/00/charronh/Desktop/CSC401/A2/"
    output_filename = "./Task5.txt"
    evalAlign(fn_LM, fn_AM, output_filename)
    
