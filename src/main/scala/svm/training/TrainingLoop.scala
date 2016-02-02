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

  val stemmer = new Stemmer_UTF8()
  stemmer.loadStemmingRules("/home/inakov/Downloads/sentiment-analysis/src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val stopWords = sc.textFile("stopwords_bg.txt").collect()

  val unigramPmiTwitterLexicon = sc.broadcast(sc.textFile("unigrams-pmilexicon-bg.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

//  val maxDiffLexicon = sc.broadcast(sc.textFile("Maxdiff-Lexicon_BG.txt")
//    .map(line => line.split("\t")).map { record =>
//    (record(0), record(1).toDouble)
//  }.collectAsMap())

  val graboLexicon = sc.broadcast(sc.textFile("grabo-pmilexicon.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val reviewsRawData = sc.textFile("reviews.csv")
  val reviewsData = reviewsRawData.map(line => line.split("~")).collect {
    case review if review.size == 4 => (review(1).toInt, review(3))
  }.filter(data => data._1.toInt != -1 && data._1.toInt != 3)


  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  val sentenceDataFrame = sqlContext.createDataFrame(reviewsData).toDF("rating", "sentence")
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
    remover.transform(regexTokenized).select("words", "filteredWords", "rating")

  import org.apache.spark.sql.functions._
  val stemmerUdf = udf { terms: Seq[String] =>
    terms.map(stemmer.stem)
  }
  val stemmedWords = filteredWords.select(col("*"), stemmerUdf(col("filteredWords")).as("stemmedWords"))

  val stemmedDocs = stemmedWords.rdd.map(row => (row(2).toString.toInt, row(3).asInstanceOf[mutable.WrappedArray[String]]))
    .filter(row => row._2.nonEmpty)

  val ngram = new NGram().setInputCol("stemmedWords").setOutputCol("ngrams")
  val ngramDataFrame = ngram.transform(stemmedWords)
  val wordSplit = ngramDataFrame.flatMap(row => List(row.getAs[mutable.WrappedArray[String]](3), row.getAs[mutable.WrappedArray[String]](4)))
    .flatMap(_.map(identity))

  val tokenCounts = wordSplit.map(t => (t, 1)).reduceByKey(_ + _)
  val tokenCountsFiltered = tokenCounts.filter{
    case (token, count) => !stopWords.contains(token) && token.length >= 2 && count >= 5
  }

  val termsDict = tokenCountsFiltered.keys.zipWithIndex().collectAsMap()
  val allTermsBroadcast = sc.broadcast(termsDict)

  val ratingsAndTokens = ngramDataFrame.select("rating", "stemmedWords", "ngrams")
    .map(row => (row.getAs[Int](0), row.getAs[mutable.WrappedArray[String]](1),
      row.getAs[mutable.WrappedArray[String]](1) ++ row.getAs[mutable.WrappedArray[String]](2)))
    .filter(_._2.nonEmpty)


  val labeledData = ratingsAndTokens.map { record =>
    val label = if (record._1 > 3) 1.0 else 0.0
    val graboLexiconFeatures = createLexiconFeatures(record._2, graboLexicon.value)
    val twitterLexiconFeatures = createLexiconFeatures(record._2, unigramPmiTwitterLexicon.value)
    val bagOfWordsFeatures = createTermsVector(record._3, allTermsBroadcast.value)
    val features = combine(Vectors.dense(graboLexiconFeatures.toArray ++ twitterLexiconFeatures.toArray).toSparse,
      bagOfWordsFeatures.toSparse)

    LabeledPoint(label, features)
  }

  def combine(v1: SparseVector, v2: SparseVector): SparseVector = {
    val size = v1.size + v2.size
    val maxIndex = v1.size
    val indices = v1.indices ++ v2.indices.map(e => e + maxIndex)
    val values = v1.values ++ v2.values
    new SparseVector(size, indices, values)
  }

  val splits = labeledData.randomSplit(Array(0.67, 0.33), seed = 11L)

  val trainingData = splits(0).cache()
  val negTest = splits(1).filter(_.label == 0)
  val posTest = splits(1).filter(_.label == 1)

  val samplingFactor = negTest.count().toDouble / posTest.count().toDouble
  val testData = negTest ++ posTest.sample(false, samplingFactor, 1234L)

  val numberOfIterations = 100
  val model = SVMWithSGD.train(trainingData, numberOfIterations)

  model.clearThreshold()

  val scoreAndLabels = testData.map { point =>
    val score = model.predict(point.features)
    val predicted = if (score > 0.5) 1 else 0
    (predicted, point.label)
  }

  val predictedPositives = scoreAndLabels.filter(_._1 == 1).count().toDouble
  val positiveCount = scoreAndLabels.filter(_._2 == 1).count().toDouble
  val correctResults = scoreAndLabels.filter(result => result._2 == 1 && result._1 == 1).count().toDouble

  val precision = correctResults/predictedPositives
  val recall = correctResults/positiveCount
  val fScore = 2*(precision*recall/(precision+recall))

  println(s"Total number of test examples: ${testData.count()}")
  println(s"Total count of positives in test data: $positiveCount")
  println(s"Precision: $precision")
  println(s"Recall: $recall")
  println(s"F-Score: $fScore")
  
}
