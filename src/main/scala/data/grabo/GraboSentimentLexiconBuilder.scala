package data.grabo

import java.io.File

import com.github.tototoshi.csv.{CSVWriter, DefaultCSVFormat}
import org.apache.spark.ml.feature.{NGram, StopWordsRemover, RegexTokenizer, Tokenizer}
import org.apache.spark.{SparkContext, SparkConf}
import svm.training.Stemmer_UTF8

import scala.collection.mutable

/**
 * Created by inakov on 16-2-1.
 */
object GraboSentimentLexiconBuilder extends App{

  val stemmer = new Stemmer_UTF8()
  stemmer.loadStemmingRules("/home/inakov/Downloads/sentiment-analysis/src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val orderingDesc = Ordering.by[(String, Int), Int](_._2)
  val orderingAsc = Ordering.by[(String, Int), Int](-_._2)
  val stopWords = sc.textFile("stopwords_bg.txt").collect()

  val reviewsRawData = sc.textFile("training-data.csv")
  val reviewsData = reviewsRawData.map(line => line.split("~")).collect {
    case review => (review(0).toInt, review(1))
  }

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

//  stemmedWords.map(row => (row(2).toString.toInt, row(3).asInstanceOf[mutable.WrappedArray[String]]))
//    .filter(_._1 < 3).flatMap(_._2.map((_, 1))).reduceByKey(_ + _).sortBy(-_._2).take(500).foreach(println)

  val stemmedDocs = stemmedWords.rdd.map(row => (row(2).toString.toInt, row(3).asInstanceOf[mutable.WrappedArray[String]]))
    .filter(row => row._2.nonEmpty).randomSplit(Array(0.67, 0.33), seed = 11L)(0).cache()

  val positiveVocab = Set("перфект", "добър", "страхот", "отлич", "препоръчва", "прекрас", "чудес", "удоволстви", "професионализ", "браво", "усмихн")
  val negativeVocab = Set("разочаров", "ужас", "лошо", "зле", "отвратител", "недовол")

  val importantWords = positiveVocab ++ negativeVocab

  val numDocuments = stemmedDocs.count().toDouble

  val termProb = stemmedDocs.map(_._2.distinct).flatMap(_.map((_, 1))).reduceByKey(_ + _)
    .map(term => (term._1, term._2.toDouble/numDocuments)).collectAsMap()

  val termProbComb = stemmedDocs.map(_._2.distinct).flatMap{document =>
    for(word <- importantWords; term <- document if document.contains(word)) yield (word+"_"+term , 1)
  }.reduceByKey(_ + _).map(term => (term._1, term._2.toDouble/numDocuments)).collectAsMap()

  def pmi(importantWord: String, term: String): Double = {
    def log2(x: Double) = scala.math.log(x)/scala.math.log(2)

    val denom = termProb(importantWord) * termProb(term)
    val combinedProb = termProbComb.getOrElse(importantWord+"_"+term, 0d)

    val pmiValue = log2(combinedProb/denom)
    pmiValue/(-log2(combinedProb))
  }

  val semanticOrientation = stemmedDocs.flatMap(_._2.distinct).distinct().map {term =>
    val positiveAssociations = {
      for(word <- positiveVocab) yield pmi(word, term)
    }.filter(!_.isNaN())

    val negativeAssociations = {
      for(word <- negativeVocab) yield pmi(word, term)
    }.filter(!_.isNaN())

    val positiveAssoc = positiveAssociations.sum/positiveAssociations.size.toDouble
    val negativeAssoc = negativeAssociations.sum/negativeAssociations.size.toDouble

    (term, positiveAssoc - negativeAssoc)
  }.filter(!_._2.isNaN).sortBy(-_._2).collect()

  implicit object MyFormat extends DefaultCSVFormat {
    override val delimiter = '\t'
  }

  val writer = CSVWriter.open(new File("/home/inakov/Downloads/sentiment-analysis/grabo-pmilexicon.txt"))

  semanticOrientation.foreach(row => writer.writeRow(List(row._1, row._2)))

  writer.close()
  println("Done")

}
