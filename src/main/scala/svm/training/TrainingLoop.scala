package svm.training

import org.apache.spark.ml.feature.{NGram, StopWordsRemover, RegexTokenizer}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vectors, Vector}

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

  val stemmer = new Stemmer_UTF8()
  stemmer.loadStemmingRules("/home/inakov/Downloads/sentiment-analysis/src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val orderingDesc = Ordering.by[(String, Int), Int](_._2)
  val orderingAsc = Ordering.by[(String, Int), Int](-_._2)
  val stopWords = sc.textFile("stopwords_bg.txt").collect()

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
    .map(row => (row.getAs[Int](0), row.getAs[mutable.WrappedArray[String]](1) ++ row.getAs[mutable.WrappedArray[String]](2)))

  val labeledData = ratingsAndTokens.map { record =>
    val label = if (record._1 > 3) 1.0 else 0.0
    LabeledPoint(label, createTermsVector(record._2, allTermsBroadcast.value))
  }

  val splits = labeledData.randomSplit(Array(0.67, 0.33), seed = 11L)

  val trainingData = splits(0).cache()
  val testData = splits(1)

  val numberOfIterations = 100
  val model = SVMWithSGD.train(trainingData, numberOfIterations)

  model.clearThreshold()

  val scoreAndLabels = testData.map { point =>
    val score = model.predict(point.features)
    val predicted = if (score > 0.5) 1 else 0
    (predicted, point.label)
  }
  val predNegCount = scoreAndLabels.filter(_._1 == 0).count().toDouble
  val negCount = scoreAndLabels.filter(_._2 == 0).count().toDouble
  val correctPredictions = scoreAndLabels.filter(result => result._2 == 0 && result._1 == 0).count().toDouble
  val prec = correctPredictions/predNegCount
  val rec = correctPredictions/negCount
  val fscore = 2*(prec*rec/(prec+rec))

  println(s"Total number of negatives: $negCount")
  println(s"Number of correct predictions: $correctPredictions")
  println(s"F-score: $fscore")

//  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//  val auROC = metrics.areaUnderROC()
//  val auPR = metrics.areaUnderPR()
//
//  println("Area under ROC = " + auROC)
//  println("Area under PR = " + auPR)
//  println(s"Training data size: ${trainingData.count()}")
//  println(s"Test data size: ${testData.count()}")

}
