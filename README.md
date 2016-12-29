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
on the [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information).
The PMI counts probability of a word to co-occurrences with positive or negative word. The positive and negative words
are manualy picked from the data set for example:

- V+ {“перфектно”, “страхотно”, “отлично”, “браво” ...}
- V- {“ужасно”, “лошо”, “зле”, “разочарован” ...}

![alt tag](https://wikimedia.org/api/rest_v1/media/math/render/svg/ff54cfce726857db855d4dd0a9dee2c6a5e7be99)

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