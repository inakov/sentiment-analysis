package svm.training

import org.apache.spark.ml.feature.{NGram, StopWordsRemover, RegexTokenizer}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{SparseVector, Vectors, Vector}

import scala.collection.{mutable, Map}

/**
 * Created by inakov on 16-1-29.
 */
object TrainingLoop extends App{

  def createTermsVector(terms: Seq[String], termDict: Map[String, Long]): Vector ={
    val featureVectorSize = termDict.size
    val indices = for(term <- terms if termDict.contains(term)) yield termDict(term)

    Vectors.sparse(featureVectorSize, indices.distinct.map(x => (x.toInt, 1.0)))
  }

  def createLexiconFeatures(terms: Seq[String], lexicon: Map[String, Double]): Vector ={
    val termsWeights = terms.map(lexicon.get).filter(_.isDefined).map(_.get)

    val positiveCount = termsWeights.count(_ >= 0).toDouble
    val negativeCount = termsWeights.count(_ < 0).toDouble
    val lastWordPolarity = lexicon.getOrElse(terms.last, 0.0)
    val lastPositiveScore = termsWeights.filter(_ >= 0).lastOption.getOrElse(0.0)
    val sumOfPositives = termsWeights.filter(_ >= 0).sum
    val sumOfNegatives = termsWeights.filter(_ < 0).sum
    val totalScore = termsWeights.sum
    val maxScore = if(termsWeights.nonEmpty) termsWeights.max else 0

    Vectors.dense(Array(positiveCount, negativeCount, lastWordPolarity,
      sumOfPositives, sumOfNegatives, totalScore, maxScore, lastPositiveScore))
  }

  def rawTextFeatures(sentance: String, emoticons: Map[String, Double]): Vector ={
    val textFeaturesRegex =  """(?:[:=;][oO\-]?[Dd\(\[\)\]\(\]/\\OpP])""".r

    val emoticionsAndPunctuation = textFeaturesRegex.findAllIn(sentance).toList
    val emoticonsFromSentance = for(ecmoticon <- emoticionsAndPunctuation if emoticons.contains(ecmoticon)) yield emoticons(ecmoticon)

    val numPositiveEmoticons = emoticonsFromSentance.count(_ == 1.0)
    val numNegativeEmoticons = emoticonsFromSentance.count(_ == -1.0)


    Vectors.dense(numPositiveEmoticons, numNegativeEmoticons)
  }

  def combine(v1: SparseVector, v2: SparseVector): SparseVector = {
    val size = v1.size + v2.size
    val maxIndex = v1.size
    val indices = v1.indices ++ v2.indices.map(e => e + maxIndex)
    val values = v1.values ++ v2.values
    new SparseVector(size, indices, values)
  }

  val stemmer = new Stemmer_UTF8()
  stemmer.loadStemmingRules("src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val stopWords = sc.textFile("src/main/resources/stopwords_bg.txt").collect()

  val unigramPmiTwitterLexicon = sc.broadcast(sc.textFile("src/main/resources/lexicons/unigrams-pmilexicon-bg.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val emoticonLexicon = sc.broadcast(sc.textFile("src/main/resources/lexicons/emoticons.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val graboLexicon = sc.broadcast(sc.textFile("src/main/resources/lexicons/grabo-pmilexicon.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val trainingRawData = sc.textFile("src/main/resources/dataset/training-data.csv")
  val trainingData = trainingRawData.map(line => line.split("~")).collect {
    case review => (review(0).toInt, review(1))
  }.cache()

  val testRawData = sc.textFile("src/main/resources/dataset/test-data.csv").map(line => line.split("~")).collect {
    case review => (review(0).toInt, review(1))
  }.cache()

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  val sentenceDataFrame = sqlContext.createDataFrame(trainingData).toDF("rating", "sentence")
  val testRawDataFrame = sqlContext.createDataFrame(testRawData).toDF("rating", "sentence")

  val regexTokenizer = new RegexTokenizer()
    .setInputCol("sentence")
    .setOutputCol("words")
    .setPattern("""[^\p{L}\p{Nd}]+""")

  val regexTokenized = regexTokenizer.transform(sentenceDataFrame)

  val remover = new StopWordsRemover()
    .setInputCol("words")
    .setOutputCol("filteredWords")
    .setStopWords(stopWords)

  val filteredWords =
    remover.transform(regexTokenized).select("words", "filteredWords", "rating", "sentence")

  val filteredTestData =
    remover.transform(regexTokenizer.transform(testRawDataFrame)).select("words", "filteredWords", "rating", "sentence")

  import org.apache.spark.sql.functions._
  val stemmerUdf = udf { terms: Seq[String] =>
    terms.map(stemmer.stem)
  }

  val stemmedWords = filteredWords.select(col("*"), stemmerUdf(col("filteredWords")).as("stemmedWords"))
  val stemmedTestData = filteredTestData.select(col("*"), stemmerUdf(col("filteredWords")).as("stemmedWords"))

  val ngram = new NGram().setInputCol("stemmedWords").setOutputCol("ngrams")
  val trainingDataFrame = ngram.transform(stemmedWords).cache()

  val testDataTokenized = ngram.transform(stemmedTestData).select("rating", "stemmedWords", "ngrams", "sentence")
    .map(row => (row.getAs[Int](0), row.getAs[mutable.WrappedArray[String]](1),
      row.getAs[mutable.WrappedArray[String]](1) ++ row.getAs[mutable.WrappedArray[String]](2), row.getAs[String](3)))
    .filter(_._2.nonEmpty).cache()
  
  val wordSplit = trainingDataFrame.flatMap(row => row.getAs[mutable.WrappedArray[String]](4) ++ row.getAs[mutable.WrappedArray[String]](5))

  val tokenCounts = wordSplit.map(t => (t, 1)).reduceByKey(_ + _)
  val tokenCountsFiltered = tokenCounts.filter{
    case (token, count) => !stopWords.contains(token) && token.length >= 2 && count >= 3
  }

  tokenCountsFiltered.saveAsObjectFile("model/grabo-vocabulary")
//  val tokenCountsFiltered = sc.objectFile[(String, Int)]("model/grabo-vocabulary")

  val termsDict = tokenCountsFiltered.keys.zipWithIndex().collectAsMap()
  val allTermsBroadcast = sc.broadcast(termsDict)

  val trainingDataTokenized = trainingDataFrame.select("rating", "stemmedWords", "ngrams", "sentence")
    .map(row => (row.getAs[Int](0), row.getAs[mutable.WrappedArray[String]](1),
      row.getAs[mutable.WrappedArray[String]](1) ++ row.getAs[mutable.WrappedArray[String]](2), row.getAs[String](3)))
    .filter(_._2.nonEmpty).cache()


  val labeledTrainingData = trainingDataTokenized.map { record =>
    val label = if (record._1 > 3) 1.0 else 0.0

    val emoticonFeatures = rawTextFeatures(record._4, emoticonLexicon.value)
    val graboLexiconFeatures = createLexiconFeatures(record._2, graboLexicon.value)
    val twitterLexiconFeatures = createLexiconFeatures(record._2, unigramPmiTwitterLexicon.value)
    val bagOfWordsFeatures = createTermsVector(record._3, allTermsBroadcast.value)
    val features = combine(
      Vectors.dense(emoticonFeatures.toArray ++ graboLexiconFeatures.toArray ++ twitterLexiconFeatures.toArray).toSparse,
      bagOfWordsFeatures.toSparse)

    LabeledPoint(label, features)
  }

  val labeledTestData = testDataTokenized.map { record =>
    val label = if (record._1 > 3) 1.0 else 0.0

    val emoticonFeatures = rawTextFeatures(record._4, emoticonLexicon.value)
    val graboLexiconFeatures = createLexiconFeatures(record._2, graboLexicon.value)
    val twitterLexiconFeatures = createLexiconFeatures(record._2, unigramPmiTwitterLexicon.value)
    val bagOfWordsFeatures = createTermsVector(record._3, allTermsBroadcast.value)
    val features = combine(
      Vectors.dense(emoticonFeatures.toArray ++ graboLexiconFeatures.toArray ++ twitterLexiconFeatures.toArray).toSparse,
      bagOfWordsFeatures.toSparse)

    LabeledPoint(label, features)
  }.cache()

  val numberOfIterations = 100
  val model = SVMWithSGD.train(labeledTrainingData, numberOfIterations)

  model.clearThreshold()

//  model.save(sc, "model/grabo-sentiment-model")

  val scoreAndLabels = labeledTestData.map { point =>
    val score = model.predict(point.features)
    val predicted = if (score > 0.5) 1 else 0
    (predicted, point.label)
  }

  val predictedNegatives = scoreAndLabels.filter(_._1 == 0).count().toDouble
  val negativeCount = scoreAndLabels.filter(_._2 == 0).count().toDouble
  val correctResults = scoreAndLabels.filter(result => result._2 == 0 && result._1 == 0).count().toDouble

  val precision = correctResults/predictedNegatives
  val recall = correctResults/negativeCount
  val fScore = 2*(precision*recall/(precision+recall))

  println(s"Total count of training examples ${labeledTrainingData.count()}")
  println(s"Total count of negatives in training data ${labeledTrainingData.filter(_.label == 0.0).count()}")
  println(s"Total count of positives in training data ${labeledTrainingData.filter(_.label == 1.0).count()}")

  println(s"Total number of test examples: ${labeledTestData.count()}")
  println(s"Total count of negatives in test data: $negativeCount")
  println(s"Total count of positives in test data: ${scoreAndLabels.filter(_._2 == 1).count()}")

  println(s"Precision: $precision")
  println(s"Recall: $recall")
  println(s"F-Score for negatives: $fScore")
  
}
