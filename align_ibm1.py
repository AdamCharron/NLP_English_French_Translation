from lm_train import *
from log_prob import *
from preprocess import *
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    sentences_e, sentences_f = read_hansard(train_dir, num_sentences)
    #for i in range(len(sentences_e)):
    #    print(sentences_e[i])
    #    print(sentences_f[i])

    if (len(sentences_e) != len(sentences_f)):
        print("ISSUE ENCOUNTERED: Mismatching lengths of english and french sentence arrays")
        print("Returned: {}, expected: {}".format(len(sentences_e),len(sentences_f)))
        #return AM
    if (len(sentences_e) != num_sentences):
        print("ISSUE ENCOUNTERED: Did not return the correct amount of sentences")
        print("Returned: {}, expected: {}".format(len(sentences_e),num_sentences))
        #return AM
    
    # Initialize AM uniformly
    AM = initialize(sentences_e, sentences_f)
    
    # Iterate between E and M steps
    for i in range(max_iter):
        print("EM Iteration: {}/{}".format(i+1, max_iter))
        AM = em_step(AM, sentences_e, sentences_f)
    
    #Save Model
    with open(fn_AM+'AM.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #print(AM)
    return AM
    
    
# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    sentences_e = []
    sentences_f = []
    sentence_count = 0
    for filename in os.listdir(train_dir):
        if filename.endswith(".e") or filename.endswith(".f"):
            name, ext = os.path.splitext(filename)
            name = os.path.join(train_dir, name)
            print(name)
            f_e = open(name + '.e', 'r')
            f_f = open(name + '.f', 'r')
            while sentence_count < num_sentences:

                # English read line
                e_line = f_e.readline()
                e_line = e_line.rstrip()
                if not e_line: break
                e_line = preprocess(e_line, 'e')
                
                # French read line
                f_line = f_f.readline()
                f_line = f_line.rstrip()
                if not f_line: break
                f_line = preprocess(f_line, 'f')
                
                # append lines to whatever it is I'm returning
                #print('\t' + e_line)
                #print('\t' + f_line)
                sentences_e.append(e_line)
                sentences_f.append(f_line)
                
                sentence_count += 1
            if sentence_count >= num_sentences: break
    return sentences_e, sentences_f

def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
	# initialize AM[‘e’] uniformly over only those French words that occur in corresponding French sentences
	# For each english word, for each sentence it appears in, for each french word in that sentence, include that word as a potential alignment
    AM = {}
    word_sum = {}
    AM['SENTSTART'] = dict([('SENTSTART', 1)])
    AM['SENTEND'] = dict([('SENTEND', 1)])
    for sent in range(len(eng)):
        #print("sent: {}".format(eng[sent]))
        eng_array = eng[sent].strip()
        eng_array = eng_array.split(' ')
        fre_array = fre[sent].strip()
        fre_array = fre_array.split(' ')
        for word in eng_array:
            if word not in word_sum: word_sum[word] = 0
            for mot in fre_array:
                #print("word: {}, mot: {}".format(word, mot))
                #print(AM)
                if word not in AM: 
                    AM[word] = dict([(mot, 1)])
                    word_sum[word] += 1
                    continue
                if mot not in AM[word]:
                    AM[word][mot] = 1
                    word_sum[word] += 1

	# Normalize to make it all uniform
    for word in word_sum:
        for mot in AM[word]:
            AM[word][mot] = 1/word_sum[word]
	
    #print("word_sum: {}".format(word_sum))
    #print("AM: {}".format(AM))
	
    return AM
    
def em_step(AM, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
	
	# Ignoring NULL words
    # Ignoring no-match-words
    
	# ======== Pseudo-code from tutorial ========
    # initialize P(f|e)                                     --> Done already in the initialize(eng, fre) function
    # for a number of iterations:                           --> Handled outside this function
        # set tcount(f, e) to 0 for all f, e
        # set total(e) to 0 for all e
        # for each sentence pair (F, E) in training corpus:
            # for each unique word f in F:
                # denom_c = 0
                # for each unique word e in E:
                    # denom_c += P(f|e) * F.count(f)
                # for each unique word e in E:
                    # tcount(f, e) += P(f|e) * F.count(f) * E.count(e) / denom_c
                    # total(e) += P(f|e) * F.count(f) * E.count(e) / denom_c
        # for each e in domain(total(:)):
            # for each f in domain(tcount(:,e)):
                # P(f|e) = tcount(f, e) / total(e)
                
    # set tcount(f, e) to 0 for all f, e
    # set total(e) to 0 for all e
    tcount = {}
    total_e = {}
    # These two are a fast way to track which words have been encountered previously for iteration later (0 = no, 1 = yes)
    e_words_found = {}
    e_words_found2 = {}
    f_words_found = {}
    for sent in range(len(eng)):
        # Split each sentence (English and French) into an array for easy iteration
        eng_array = eng[sent].strip()
        eng_array = eng_array.split(' ')
        fre_array = fre[sent].strip()
        fre_array = fre_array.split(' ')
        
        # Iterate through each word in English and French and zero the lookup counters
        for word in eng_array:
            if word not in total_e: total_e[word] = 0
            if word not in e_words_found: e_words_found[word] = 0
            for mot in fre_array:
                #print("word: {}, mot: {}".format(word, mot))
                if word not in tcount: 
                    tcount[word] = dict([(mot, 0)])
                else:
                    tcount[word][mot] = 0
                if mot not in f_words_found: f_words_found[mot] = 0
                
    # for each sentence pair (F, E) in training corpus:
    for sent in range(len(eng)):
        #print("Sentence: {}".format(sent+1))
        # Split each sentence (English and French) into an array for easy iteration
        eng_array = eng[sent].strip()
        eng_array = eng_array.split(' ')
        fre_array = fre[sent].strip()
        fre_array = fre_array.split(' ')
        
        # Reset found check
        for mot in f_words_found: f_words_found[mot] = 0
        
        # for each unique word f in F:
        for mot in fre_array:
            f_words_found[mot] += 1
            if f_words_found[mot] > 1: continue
            denom_c = 0
            
            # Reset found check
            for word in e_words_found: e_words_found[word] = 0
            e_words_found2 = dict(e_words_found)
            
            # for each unique word e in E:
            for word in eng_array:
                e_words_found[word] += 1
                if e_words_found[word] > 1: continue
                #denom_c += P(f|e) * F.count(f)
                denom_c += AM[word][mot]*f_words_found[mot]

            # for each unique word e in E:
            for word in eng_array:
                e_words_found2[word] += 1
                if e_words_found2[word] > 1: continue
                # increment_val = P(f|e) * F.count(f) * E.count(e) / denom_c
                # tcount(f, e) += increment_val    # Slide 22-23
                # total(e) += increment_val
                increment_val = (AM[word][mot] * f_words_found[mot] * e_words_found[word]) / denom_c
                tcount[word][mot] += increment_val
                total_e[word] += increment_val
                #print("word: {}, mot: {}, tcount[word][mot]: {}, total_e[word]: {}".format(word, mot, tcount[word][mot], total_e[word]))
                
    # Maximization Step
    # for each e in domain(total(:)):
    for word in total_e:
        # for each f in domain(tcount(:,e)):
        for mot in tcount[word]:
            # P(f|e) = tcount(f, e) / total(e)
            AM[word][mot] = tcount[word][mot]/total_e[word]

    return AM



