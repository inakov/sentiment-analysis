package svm.training

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import scala.collection.Map

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
  stemmer.loadStemmingRules("/home/inakov/IdeaProjects/sentiment-analysis/src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val orderingDesc = Ordering.by[(String, Int), Int](_._2)
  val orderingAsc = Ordering.by[(String, Int), Int](-_._2)
  val stopWords = sc.textFile("stopwords_bg.txt").collect().toSet

  val reviewsRawData = sc.textFile("reviews.csv")
  val reviewsData = reviewsRawData.map(line => line.split("~")).collect {
    case review if review.size == 4 => (review(0).toInt, review(1).toInt, review(2).toInt, review(3))
  }.filter(_._2 != -1)


  val text = reviewsData.map(reviewData => reviewData._4)
  val nonWordSplit = text.flatMap(t => t.split("""[^\p{L}\p{Nd}]+""").map(_.toLowerCase).map(stemmer.stem))
  val tokenCounts = nonWordSplit.map(t => (t, 1)).reduceByKey(_ + _)
  val tokenCountsFiltered = tokenCounts.filter{
    case (token, count) => !stopWords.contains(token) && token.length >= 2 && count >= 2
  }

  val termsDict = tokenCountsFiltered.keys.zipWithIndex().collectAsMap()
  val allTermsBroadcast = sc.broadcast(termsDict)

  val ratingAndTokens = reviewsData.map(reviewData => (reviewData._2, reviewData._4))
    .map(r => (r._1, r._2.split("""[^\p{L}\p{Nd}]+""").map(_.toLowerCase).map(stemmer.stem))).filter(_._2.nonEmpty)


//  ratingAndTokens.filter(_._1 >= 4).flatMap(_._2.map(s => (s, 1))).reduceByKey(_ + _).filter{
//    case (token, count) => !stopWords.contains(token) && token.length >= 2
//  }.sortBy(-_._2).take(100).foreach(println)
//
  ratingAndTokens.filter(_._1 < 4).flatMap(_._2.map(s => (s, 1))).reduceByKey(_ + _).filter{
    case (token, count) => !stopWords.contains(token) && token.length >= 2
  }.sortBy(-_._2).take(100).foreach(println)
  val labeledData = ratingAndTokens.map { record =>
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
    (score, point.label)
  }

  val metrics = new BinaryClassificationMetrics(scoreAndLabels)
  val auROC = metrics.areaUnderROC()
  val auPR = metrics.areaUnderPR()

  println("Area under ROC = " + auROC)
  println("Area under PR = " + auPR)
  println(s"Training data size: ${trainingData.count()}")
  println(s"Test data size: ${testData.count()}")

}
