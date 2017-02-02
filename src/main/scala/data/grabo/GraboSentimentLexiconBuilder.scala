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
  stemmer.loadStemmingRules("src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val orderingDesc = Ordering.by[(String, Int), Int](_._2)
  val orderingAsc = Ordering.by[(String, Int), Int](-_._2)
  val stopWords = sc.textFile("src/main/resources/stopwords_bg.txt").collect()

  val reviewsRawData = sc.textFile("src/main/resources/dataset/training-set.csv")
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

  val tokenFreq = stemmedWords.map(row => (row(2).toString.toInt, row(3).asInstanceOf[mutable.WrappedArray[String]]))
    .flatMap(review => review._2.map{token=>
      if (review._1 > 3) (token, (1, 0))
      else (token, (0, 1))
    }).reduceByKey((f1, f2 )=> (f1._1 + f2._1, f1._2 + f2._2))
    .filter(token => token._2._1 + token._2._2 > 5)

  val (posFreq, negFreq) = tokenFreq.map(_._2).reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
  val totalFreq = posFreq + negFreq

  def sentimentScore(freqWPositive: Double,
                     freqWNegative: Double): Double = {

    def pmi(freqWSentiment: Double, tokenCount: Double, freqW: Double, freqSentiment: Double): Double = {
      def log2(x: Double) = scala.math.log(x)/scala.math.log(2)
      log2((freqWSentiment*tokenCount)/(freqW*freqSentiment))
    }
    val totalWFreq = freqWPositive + freqWNegative

    val posPmi = if(freqWPositive != 0){
      pmi(freqWPositive, totalFreq, totalWFreq, posFreq)
    } else 0.0

    val negPmi = if(freqWNegative != 0){
      pmi(freqWNegative, totalFreq, totalWFreq, negFreq)
    }else 0.0

    posPmi - negPmi
  }

  val sentimentOrientation =
    tokenFreq.map(token => (token._1, sentimentScore(token._2._1, token._2._2)))
    .sortBy(-_._2).collect()

  implicit object MyFormat extends DefaultCSVFormat {
    override val delimiter = '\t'
  }

  val writer = CSVWriter.open(new File("/home/inakov/GitHub/sentiment-analysis/src/main/resources/lexicons/grabo-pmilexicon.txt"))
  sentimentOrientation.foreach(row => writer.writeRow(List(row._1, row._2)))
  writer.close()
  println("Done")
}
