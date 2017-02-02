Sentiment analysis
==================

The purpose of the project is to determine whether a given review/comment contains positive or negative opinion based 
only on text features. The dataset contains raw unfiltered reviews from popular group shopping platform in Bulgaria.

Text Mining
-----------

The raw data from the comments is tokenized by common text processing patterns. 
After the tokenization, a simple model called [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) 
is applied to extract raw text features.

Tokenization technique

- Break down the text stream into words
- Removal of "Stopwords" i.e. (ако, и, в, ала ...)
- Stemming

Bag of words

- Sparse Vectors of n-grams

Semantic orientation lexicon
----------------------------

We generated a [lexicon](https://github.com/inakov/sentiment-analysis/blob/master/src/main/resources/lexicons/grabo-pmilexicon.txt)
with the semantic orientation of common words from the data set based
on the Sentiment Score of the tokens in the training set.

SentimentScore(w) = PMI(w, positive) - PMI(w, negative)

PMI stands for [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information).

_PMI(w, positive) = log2((freq(w, positive) * N )/(freq(w) * freq(positive)))_

_freq(w, positive)_ is the number of times a term _w_ occurs in a positive review, _freq(w)_ is 
the total frequency of term _w_, _freq(positive)_ is the total number of tokens in positive reviews and 
_N_ is the total number of tokens.

Classifier and Features
-----------------------

For classification we are using Apache Spark's implementation of
[Support Vector Machine (SVM)](https://spark.apache.org/docs/latest/mllib-linear-methods.html#linear-support-vector-machines-svms)

We selected the following features for our model

- Sparse Vectors of n-grams
- Count of words with positive orientation(SO > 0)
- Count of words with negative orientation(SO < 0)
- Sum of all positive words
- Sum of all negative words
- The highest SO of a word in the comment
- SO of the last word in the comment
- Count of positive emoticons.
- Count of negative emoticons.

Results
-------

To measure the performance of our model we are using
[F1-Score](https://en.wikipedia.org/wiki/F1_score).

Our F1-Score is 0.8453006968