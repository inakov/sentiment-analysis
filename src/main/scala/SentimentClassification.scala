import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.{SparseVector, Vectors, Vector}
import org.apache.spark.{SparkContext, SparkConf}
import svm.training.Stemmer_UTF8

import scala.collection.Map
import scala.io.StdIn

/**
 * Created by inakov on 16-2-4.
 */
object SentimentClassification extends App{

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
  stemmer.loadStemmingRules("/home/inakov/Downloads/sentiment-analysis/src/main/resources/stem_rules_context_2_UTF-8.txt")

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val stopWords = sc.textFile("stopwords_bg.txt").collect()

  val unigramPmiTwitterLexicon = sc.broadcast(sc.textFile("unigrams-pmilexicon-bg.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val emoticonLexicon = sc.broadcast(sc.textFile("emoticons.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val graboLexicon = sc.broadcast(sc.textFile("grabo-pmilexicon.txt")
    .map(line => line.split("\t")).map { record =>
    (record(0), record(1).toDouble)
  }.collectAsMap())

  val tokenCountsFiltered = sc.objectFile[(String, Int)]("model/grabo-vocabulary")

  val termsDict = tokenCountsFiltered.keys.zipWithIndex().collectAsMap()
  val allTermsBroadcast = sc.broadcast(termsDict)

  val sentimentModel = SVMModel.load(sc, "model/grabo-sentiment-model")

  while(true){
    println("Моля, въведете вашият коментар:")
    val comment = StdIn.readLine()

    val tokens = comment.split("""[^\p{L}\p{Nd}]+""").map(_.toLowerCase)
      .filterNot(stopWords.contains(_)).map(stemmer.stem)

    val bigrams = tokens.sliding(2).map(p => p.mkString(" "))


    val emoticonFeatures = rawTextFeatures(comment, emoticonLexicon.value)
    val graboLexiconFeatures = createLexiconFeatures(tokens, graboLexicon.value)
    val twitterLexiconFeatures = createLexiconFeatures(tokens, unigramPmiTwitterLexicon.value)
    val bagOfWordsFeatures = createTermsVector(tokens ++ bigrams, allTermsBroadcast.value)
    val features = combine(
      Vectors.dense(emoticonFeatures.toArray ++ graboLexiconFeatures.toArray ++ twitterLexiconFeatures.toArray).toSparse,
      bagOfWordsFeatures.toSparse)

    val score = sentimentModel.predict(features)

    print("Вашият коментар беше оценен като: ")
    if (score > 0.5) println("+ Положителен")
    else println(" - Отрицателен")
  }

}
