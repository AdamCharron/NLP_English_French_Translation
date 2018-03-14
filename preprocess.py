import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 
	
	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence
				   
	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    
    # Separate: 
    #   sentence-final punctuation (sentences have already been determined for you), 
    #   commas,
    #   colons and semicolons,
    #   parentheses, 
    #   dashes between parentheses, 
    #   mathematical operators (e.g., +, -, <, >, =), 
    #   and quotation marks.    
    out_sentence = in_sentence.strip()
    out_sentence = re.sub('([\.,:;\(\)\+\-\<\>\=`\"])', r' \1 ', out_sentence)          # Separate most punctuation and mathematical operators
    out_sentence = re.sub('(.+)([\.?:,!\-\(\)\[\]])(\s)*$', r'\1 \2', out_sentence)     # Remove end-line punctuation
    out_sentence = re.sub('(\'{2,})', r' \1 ', out_sentence)                            # Separate multiple consecutive apostrophes (pseudo-quotations)
    out_sentence = re.sub(' +', ' ', out_sentence.strip())                              # Remove excess whitespace
    
    # Convert all tokens to lower-case
    out_sentence = out_sentence.lower()
    
    if (language == 'f'):
        # When the language is french:
        #   Separate leading l' from concatenated word
        #   Separate leading qu' from concatenated word
        #   Separate leading consonant and apostrophe from concatenated word
        out_sentence = re.sub('(\W+)(qu|b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|z|)(\')(\w+)', r'\1\2\3 \4', out_sentence)
        out_sentence = re.sub('^(qu|b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|z|)(\')(\w+)', r'\1\2 \3', out_sentence)
        
        #   Separate following on or il
        out_sentence = re.sub('(\w+)(\')(on|il)([^a-zA-Z])', r'\1\2 \3\4', out_sentence)
        out_sentence = re.sub('(\w+)(\')(on|il)$', r'\1\2 \3', out_sentence)
    
    return out_sentence
