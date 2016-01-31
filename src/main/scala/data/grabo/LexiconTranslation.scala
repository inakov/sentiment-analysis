package data.grabo

import java.io.File

import com.github.tototoshi.csv.{CSVWriter, DefaultCSVFormat}
import org.apache.spark.{SparkContext, SparkConf}
import svm.training.Stemmer_UTF8

/**
 * Created by inakov on 31.01.16.
 */
object LexiconTranslation extends App{

  val stemmer = new Stemmer_UTF8()
  stemmer.loadStemmingRules("/home/inakov/IdeaProjects/sentiment-analysis/src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Lexicon translation")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val lexiconData = sc.textFile("/home/inakov/Documents/Sentiment140-Lexicon-v0.1/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon-clean.txt")
    .map(line => line.split("\t")).map(line => (line(0), line(1), line(2), line(3)))
  val translationData = sc.textFile("/home/inakov/Documents/Sentiment140-Lexicon-v0.1/Sentiment140-Lexicon-v0.1/translated.txt")

  val zippedData = lexiconData.zipWithIndex().map(t=> (t._2, t._1))
    .join(translationData.zipWithIndex().map(t=> (t._2, t._1))).values

  val resultLexicon = zippedData.map(line=> (line._2, line._1._2, line._1._3, line._1._4)).filter{
    _._1.trim.nonEmpty
  }.map(line=> (stemmer.stem(line._1.toLowerCase), line._2, line._3, line._4)).sortBy(-_._2.toDouble).collect()

  implicit object MyFormat extends DefaultCSVFormat {
    override val delimiter = '\t'
  }

  val writer = CSVWriter.open(new File("/home/inakov/IdeaProjects/sentiment-analysis/unigrams-pmilexicon-bg.txt"))

  resultLexicon.foreach(row => writer.writeRow(List(row._1, row._2, row._3, row._4)))

  writer.close()
  println("Done")
}
