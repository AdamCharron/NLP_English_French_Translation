from decode import *
from BLEU_score import *
from preprocess import *
from lm_train import *
from align_ibm1 import *
import os
import math

def evalAlign(fn_LM, fn_AM, output_filename):
    """
	Produces translations from french to English, obtains reference translations from Google and the Hansards, and use the latter to evaluate the former, with a BLEU score
	
	INPUTS:
	fn_LM : 		    (string) the location of the the language model
	fn_AM : 		    (string) the location to save the alignment model
	output_filename :	(string) the location and filename of the output file (ex: path/Task5.txt)

	
	OUTPUT:
	None - writes to a file output_filename instead
	"""

    # For evaluation, translate the 25 French sentences in /u/cs401/A2_SMT/data/Hansard/Testing/Task5.f with the decode function and evaluate them using corresponding reference sentences, specifically:
    #   1. /u/cs401/A2_SMT/data/Hansard/Testing/Task5.e, from the Hansards.
    #   2. /u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e, Google’s translations of the French phrases
    # Test files
    f_test_file = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.f"
    e_test_file = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.e"
    google_e_test_file = "/u/cs401/A2_SMT/data/Hansard/Testing/Task5.google.e"
    
    # Training directory and target save directories
    train_dir = "/u/cs401/A2_SMT/data/Hansard/Training/"
    #train_dir = "/h/u16/c8/00/charronh/Desktop/CSC401/A2/"
    
    # Training English model from part 2
    language = 'e'
    LM = lm_train(train_dir, language, fn_LM)

    # Set up output file    
    out_file = open(output_filename, 'w')
    out_file.write("n,\tnumber_sentences,\tBLEU Score,\t\t\t\t#Line:\tOutputted English sentence\n")

    
    # Training
    n_vals = [1,2,3]
    num_sentence_array = [1000, 10000, 15000, 30000]
    
    # Pre-train the models to save tons of time
    AM_array = []
    max_iter = 5
    for i in range(len(num_sentence_array)):
        print("Training AM for max_iter: {} with # of sentences: {}".format(max_iter, num_sentence_array[i]))
        temp_AM = align_ibm1(train_dir, num_sentence_array[i], max_iter, fn_AM)
        AM_array.append(temp_AM)
        # As per assignment specifications, save the model for 1k number of sentences
        if num_sentence_array[i] == 1000:
            with open('am.pickle', 'wb') as handle:
                pickle.dump(AM_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Handle decoding and BLEU score evaluation
    for n in n_vals:
        for i in range(len(num_sentence_array)):
            print("n: {}, num_sentences: {}".format(n, num_sentence_array[i]))
            
            # Get preprocessed french lines for testing 
            f_f = open(f_test_file, 'r')
            f_google_e = open(google_e_test_file, 'r')
            f_e = open(e_test_file, 'r')
            
            line_count = 0
            while True:
                line_count += 1
                # English read line
                e_line = f_e.readline()
                e_line = e_line.rstrip()
                if not e_line: break
                e_line = preprocess(e_line, 'e')
                
                # Google English read line
                e_google_line = f_google_e.readline()
                e_google_line = e_google_line.rstrip()
                if not e_google_line: break
                e_google_line = preprocess(e_google_line, 'e')
                
                # French read line
                f_line = f_f.readline()
                f_line = f_line.rstrip()
                if not f_line: break
                f_line = preprocess(f_line, 'f')
            
                #print("")
                #print("English:\t{}".format(e_line))
                #print("Google E:\t{}".format(e_google_line))
                #print("French:\t\t{}".format(f_line))

                decoded_eng = decode(f_line, LM, AM_array[i])
                #print(decoded_eng)
                
                #bleu_score = BLEU_score(candidate, references, n)
                bleu_score = BLEU_score(decoded_eng, [e_line, e_google_line], n)
                out_file.write("{},\t{},\t\t\t\t{},\t\t\t\t\t{}:\t{}\n".format(n, num_sentence_array[i], bleu_score, line_count, decoded_eng))
            
            # Close files
            f_f.close()
            f_google_e.close()
            f_e.close()
                
    out_file.close()
    
    
