from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM
	
	INPUTS:
	
    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained
    
    OUTPUT
	
	LM			: (dictionary) a specialized language model
	
	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which 
	incorporate unigram or bigram counts
	
	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """
    # Instantiate the empty dictionaries for 'uni' and 'bi' grams
    language_model = {'uni':dict(), 'bi':dict()}
    
    # Make sure the directory exists
    if not (os.path.isdir(data_dir) or os.path.exists(data_dir)): 
        print("Directory: {} does not exist".format(data_dir))
        return None
    
    # Iterate through each file in the directory
    # Only select files with extensions matching the language selected ('e' or 'f')
    for filename in os.listdir(data_dir):
        if filename.endswith('.' + language):
            #print(filename)
            f = open(os.path.join(data_dir, filename), 'r')
            # For each of these files, read each line in the file
            # Pass each line through the preprocess function from Task 1
            # Split the resulting preprocessed string into an array (split by spaces)
            # Also add SENTSTART and SENTEND to the list of available words
            for line in f:
                line = line.rstrip()
                pre_proc_line = preprocess(line, language)
                line_arr = ['SENTSTART']
                line_arr += pre_proc_line.split(' ')
                line_arr += ['SENTEND']
                prev_word = None
                for word in line_arr:
                    # Increment counters for unigram and bigram occurrances of each word/pair of successive words encountered
                    # Need to track the previous word as well as the current one to do this
                    # Also requires populating the dictionary, so need to handle adding vs accessing keys in nested dicts (the if statements below)
                    if word in language_model['uni']:
                        language_model['uni'][word] += 1
                    else:
                        language_model['uni'][word] = 1
                    if prev_word != None:
                        if prev_word in language_model['bi']:
                            if word in language_model['bi'][prev_word]:
                                language_model['bi'][prev_word][word] += 1
                            else:
                                language_model['bi'][prev_word][word] = 1
                        else:
                            language_model['bi'][prev_word] = dict([(word, 1)])
                    prev_word = word
    
    # Print statements for debugging
    #from pprint import pprint                    
    #pprint(language_model['uni'])
    #pprint(language_model['bi'])

    #Save Model
    with open(fn_LM+language+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return language_model
