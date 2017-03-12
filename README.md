# Sentence Completion Using Pointwise Mutual Information

Implementation of the pointwise mutual information (PMI) model described in "Exploiting Linguistic Features for Sentence Completion".

If you are using this code in your work, please cite the following publication:

Woods, Aubrie M. "Exploiting Linguistic Features for Sentence Completion." In The 54th Annual Meeting of the Association for Computational Linguistics, p. 438. 2016.

```latex
@inproceedings{woods2016,
  title={Exploiting Linguistic Features for Sentence Completion},
  author={Woods, Aubrie M.},
  booktitle={The 54th Annual Meeting of the Association for Computational Linguistics},
  pages={438},
  year={2016}
}
```

For demonstration purposes, all files contain only the information needed to operate on a small subset of the Microsoft Research Sentence
Completion Challenge; specifically, the first ten questions from the test set.  Note that the code is compatible with Python 2.7.13.


## Running the code

python main.py


## Input

File  | Description
------------- | -------------
CSV files  | The unigram, bigram, and trigram co-occurrence matrices
parts_of_speech.txt  | Output from the Stanford POS tagger
dependencies.txt | Word relationships identified by the Stanford dependency parser 
keywords.txt | Common nouns identified by the Stanford POS tagger
msr_questions.txt | Microsoft Research Sentence Completion Challenge questions 521-530 in machine format
msr_answers.txt | Answers to questions 521-530 in machine format



