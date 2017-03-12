from matrix import Matrix
from pmi import pmi
from itertools import *
import re
import csv
from copy import deepcopy
from operator import itemgetter
import numpy as np
from numpy import array, dot, diag
from numpy.linalg import svd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from scipy import spatial
import requests


QUESTIONS = "input/msr_questions.txt"
ANSWERS = "input/msr_answers.txt"
KEYWORDS = "input/keywords.txt"
DEPENDENCIES = "input/dependencies.txt"
POS = "input/parts_of_speech.txt"
UNIGRAMS = 'input/unigrams.csv'
BIGRAMS = 'input/bigrams.csv'
TRIGRAMS = 'input/trigrams.csv'
RESULTS = 'output/results.csv'
PUNCTUATION = (';', ':', ',', '.', '!', '?','(',')',"'", '~')
unigram_scores = {}
unigram_bigram_scores = {}


def get_stop_words(sentence):
     """Identifies which words to remove based on their part-of-speech tags."""
     pos_to_remove = ('dt','prp','prp$','cc','nnp','nnps')
     words_to_remove = []
     tokens = sentence.strip().split(" ")
     for word_tag in tokens:
          words = word_tag.split("_")
          tokens = nltk.word_tokenize(words[0])
          t = tokens[0].lower().strip()
          tokens = nltk.word_tokenize(words[1])
          pos = tokens[0].lower().strip()
          if (pos in pos_to_remove):
               words_to_remove.append(t) 
     return words_to_remove
    
    
def get_dependencies(option, sentence):
     """Returns sentence words that share a relationship with the missing word."""
     retained_words = set()
     tokens = sentence.strip().split(" ")
     for word_tag in tokens:
          words = word_tag.split("_")
          tokens = nltk.word_tokenize(words[0])
          first_word = tokens[0].lower().strip()
          tokens = nltk.word_tokenize(words[1])
          second_word = tokens[0].lower().strip()
          if(first_word == option or second_word == option):
               if(first_word != option and first_word != "null"):
                    retained_words.add(first_word)
               if(second_word != option and second_word != "null"):
                    retained_words.add(second_word)
     return list(retained_words)
    
    
def score_option(p, candidate_words, features):
     """Computes the PMI score for a particular candidate answer."""
     candidate_score = 0.0
     for c in candidate_words:
          for f in features:    
               feature_word_index = p.get_word_index(f)
               candidate_word_index = p.get_word_index(c)
               if (feature_word_index != -1 and candidate_word_index != -1):
                    pmi = p.get_value(feature_word_index, candidate_word_index)
                    candidate_score += pmi              
     return candidate_score


def get_candidate_ngrams(candidate_word, ngrams): 
     """Replaces blanks with the candidate word."""
     candidate_options = []
     for n in ngrams:
          if(re.findall("~", n)):
               result = re.sub("~", candidate_word, n)  
               candidate_options.append(result)
     return candidate_options 
    
    
def get_keywords(ngram_type, word_list):
     """Returns the keywords associated with the specified n-gram type."""
     keywords = []
     words = word_list.split(",")
     for w in words:
          tokens = nltk.word_tokenize(w)
          if (ngram_type == 'unigram'):
               for t in tokens:
                    keywords.append(t.lower().strip())
          else:
               ngrams = get_ngrams(tokens, ngram_type) 
               for n in ngrams:
                    keywords.append(n)
     return keywords
    

def get_ngrams(tokens, ngram_type):
     """Builds n-grams from the list of tokens."""
     ngrams = []
     if (ngram_type == 'bigram'):
          words = zip(tokens, tokens[1:])
     elif (ngram_type == 'trigram'):
          words = zip(tokens, tokens[1:], tokens[2:])
     for ngram in words:
          result = ""
          for w in ngram:
              result += w.lower().strip() + "_"
          result = result[:-1]
          ngrams.append(result)
     return ngrams
     
     
def get_tokens(text, ngram_type):
     """Returns the set of tokens in a sentence."""
     words = []
     tokens = nltk.word_tokenize(text)   
     for t in tokens:
          t = t.strip().lower()
          if (ngram_type == 'unigram'):
               #remove punctuation and duplicate words
               if (t not in PUNCTUATION and t not in words):
                    words.append(t)
          else:
               #all tokens are needed for valid n-grams
               words.append(t)
     return words
     
     
def evaluate(p, ngram_type, questions, answers, options, dependencies, parts_of_speech, keywords):
     """Displays the accuracy achieved using features associated with the each n-gram type."""
     guesses = []
     pmi_scores = []
     if(ngram_type == 'unigram'):
          results_file = open(RESULTS, 'w')
          results_file = csv.writer(results_file, delimiter=',')
          results_header = ['Best Guess', 'Correct Answer', 'PMI Score', 'Question', 'Reduced Context', 'Dependencies', 'Keywords']
          results_file.writerow(results_header)
     for i in range(0, len(questions)): 
          choice = options[i]
          features = get_tokens(questions[i], ngram_type) 
          reduced_context = deepcopy(features)
          if (ngram_type != 'unigram'):
               features = get_ngrams(features, ngram_type)
          returned_keywords = get_keywords(ngram_type, keywords[i][0])  
          features += returned_keywords
          if(ngram_type == 'unigram'):
               dependent_words = get_dependencies(choice[0], dependencies[i][0])
               features += dependent_words
               words_to_remove = get_stop_words(parts_of_speech[i][0]) 
               reduced_context = filter(lambda x: x not in words_to_remove, reduced_context)
               features = filter(lambda x: x not in words_to_remove, features)
      
          best_guess = ""
          highest_score = 0.0    
          scores = []  
          for j in range(0, len(choice)):
               candidate_word = choice[j]
               if (ngram_type == 'unigram'):
                    ngram_score = score_option(p, [candidate_word], features)
               elif (ngram_type == 'bigram'):
                    candidate_options = get_candidate_ngrams(candidate_word, features)
                    ngram_score = score_option(p, candidate_options, features)     
                    ngram_score += unigram_scores[i][j]
               else:
                    candidate_options = get_candidate_ngrams(candidate_word, features)
                    ngram_score = score_option(p, candidate_options, features)   
                    ngram_score += unigram_bigram_scores[i][j]                  
               scores.append(ngram_score)
               if (ngram_score > highest_score):
                    highest_score = ngram_score
                    best_guess = candidate_word
               if (ngram_type == 'unigram'):
                    unigram_scores[i] = scores
               if (ngram_type == 'bigram'):
                    unigram_bigram_scores[i] = scores     
          if (ngram_type == 'unigram'):
               pmi_scores.append(highest_score)
               results_file.writerow([best_guess, answers[i], highest_score, questions[i], " ".join(reduced_context), " ".join(dependent_words), " ".join(returned_keywords)])          
          guesses.append(best_guess)  
     print "Accuracy: " + str(accuracy_score(answers, guesses))   
     
  
def load_matrix(path):
     """Creates a Matrix from the CSV file located at the provided path."""
     filename = path.split("/")[-1]
     print "Reading " + filename + "..."
     return Matrix(path)


def include_ngram(ngram_type, questions, answers, options, dependencies, parts_of_speech, keywords):
     """Incorporates the specified ngram type into the semantic similarity assessment."""
     cooccurrences = "input/" + ngram_type + "s.csv"
     m = load_matrix(cooccurrences)
     print "Computing " + ngram_type + " PMI matrix..."
     p = pmi(m, positive=True, discounting=True)
     evaluate(p, ngram_type, questions, answers, options, dependencies, parts_of_speech, keywords)

          
def read_file(filename):
     """Retrieves the contents of a machine-formatted file."""
     questions = []
     with open(filename, 'r') as input_file:
        while True:
            next_n_lines = list(islice(input_file, 5))
            if not next_n_lines:
                break
            #process next_n_lines
            lines = []
            for l in next_n_lines:
                lines.append(l.lower().strip())
            questions.append(lines)
     input_file.close()
     return questions   
     
     
def read_answers(filename):
    """Returns an ordered list of answers."""
    answers = []
    file = open(filename, 'r') 
    for line in file:
        a = re.search('\[(.+)\]', line).group(1) #extract answer
        a = a.lower().strip()
        answers.append(a)
    file.close()
    return answers

    
def read_questions(filename):
     """Returns an ordered list of questions and each question's corresponding list of candidate answers."""
     options = [] #candidate words
     questions = []
     contents = read_file(filename)
     for c in contents:
          local_options = []
          for sentence in c:
               candidate_word = re.search('\[(.+)\]', sentence).group(1) #extract candidate word
               local_options.append(candidate_word)
          options.append(local_options)
          text = re.sub('\d{1,4}[a-z]\) ', '', sentence) #remove question number  
          text = re.sub('\[(.+)\]', '~', text) #replace option with blank
          #Remove period from common abbreviations
          text = re.sub('Mr\.', 'Mr', text)
          text = re.sub('Mrs\.', 'Mrs', text)
          text = re.sub('Dr\.', 'Dr', text)
          text = text.lower().strip() #remove newline character      
          questions.append(text)
     return questions, options

    
if __name__ == "__main__":  
     questions, options = read_questions(QUESTIONS)
     answers = read_answers(ANSWERS)
     dependencies = read_file(DEPENDENCIES)
     parts_of_speech = read_file(POS)
     keywords = read_file(KEYWORDS)
     include_ngram('unigram', questions, answers, options, dependencies, parts_of_speech, keywords)
     include_ngram('bigram', questions, answers, options, dependencies, parts_of_speech, keywords)
     include_ngram('trigram', questions, answers, options, dependencies, parts_of_speech, keywords)
    
     