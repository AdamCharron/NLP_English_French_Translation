import math

def BLEU_score(candidate, references, n):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing
	
	INPUTS:
	candidate :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
    
    bleu_score = 0
    cand_array = candidate.strip().split()
    N = len(cand_array)     # N is the number of words in the candidate

    # BPc = brevity    
    BPc = 0
    best_r = float("inf")
    for ref in references:
        #print("Reference: {}".format(ref))
        r = len(ref.strip().split())
        if abs(r - N) < abs(best_r - N):
            best_r = r
    if best_r < N:
        BPc = 1
    else:
        BPc = math.exp(1 - best_r/N)


    # N-gram Precision (p1*p2*p3)
    # Set up p-values for each of the n-gram values (1,2,3), and a product p_val for use later
    p = [0 for i in range(n)]
    p_val = 1
    
    # Iterate through each n-gram value
    for i in range(n):
        # C is the number of words in the candidate which are in at least one reference (becomes string of n words in n-gram examples with n>1)
        C = 0
        
        # Build a string of the next n words
        for j in range(len(cand_array)):
            check_array = []
            for k in range(i+1):
                if j + k >= len(cand_array): 
                    check_array = []
                    break
                check_array.append(cand_array[j+k])
            check_str = ' '.join(check_array)
            # Using this created string, check to see if those words are in any of the references
            for ref in references:
                if check_str in ref:
                    C += 1
                    break

        # Compute the N-gram precision p = C/N
        if N == 0: 
            p[i] = 0
        else: 
            p[i] = C/N
        
        # Multiply each of the p values (for each n) together for a product p_val
        p_val *= p[i]        
        #print("n: {}, C: {}, N: {}, p[i]: {}, p_val: {}".format(i+1, C, N, p[i], p_val))
    

    # Get and return final BLEU score
    bleu_score = BPc*(p_val)**(1/n)
    
    return bleu_score
