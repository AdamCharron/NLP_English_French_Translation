from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary
	
	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
	
    # Convert the sentence into an array form, including SENTSTART and SENTEND
    # Set up other variables before iterating
    log_prob = 0
    sentence = sentence.strip()
    sentence_array = ["SENTSTART"]
    sentence_array += sentence.split(' ')
    sentence_array += ["SENTEND"]
    prev_word = None
    
    if smoothing:
        # Delta-smoothing estimate of the sentence
        # P(wt|wt−1;δ,‖V‖) = (Count(wt−1,wt) +δ) / (Count(wt−1) +δ‖V‖).
        for word in sentence_array:
            if prev_word != None:
                num = 0
                den = 0
                if prev_word in LM['bi'] and word in LM['bi'][prev_word]: num = LM['bi'][prev_word][word]
                if prev_word in LM['uni']: den = LM['uni'][prev_word]
                log_prob += log((num + delta)/(den + delta*vocabSize), 2)
            prev_word = word
    else:
        # MLE of the sentence
        # P(wt|wt-1) = Count(wt−1,wt)/Count(wt−1)
        for word in sentence_array:
            if prev_word != None:
                if prev_word not in LM['uni'] or prev_word not in LM['bi'] or word not in LM['bi'][prev_word]:
                    return float('-inf')
                log_prob += log(LM['bi'][prev_word][word]/LM['uni'][prev_word], 2)
            prev_word = word
            
    return log_prob
