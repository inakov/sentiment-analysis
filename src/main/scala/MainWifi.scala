import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{Statistics, MultivariateStatisticalSummary}
import org.apache.spark.{SparkContext, SparkConf}
import org.joda.time.DateTime

/**
 * Created by inakov on 16-4-6.
 */

object MainWifi extends App{

  val conf = new SparkConf().setAppName("Sentiment Analysis - SVM Training Loop")
    .setMaster("local[4]").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)

  val rawData = sc.textFile("/home/inakov/Documents/trace-v3-10feed900252-2016-04-05.csv")
  val data = rawData.map(line => line.split(",")).collect {
    case check => ((check(0).toDouble * 1000).toLong, check(2), check(3), check(4), check(5))
  }.filter(_._5 == "STA").cache()

//  data.take(100).foreach(println)
//  val countByUser = data.map(check => (check._2, 1)).reduceByKey(_ + _).sortBy(-_._2)
  val groupByUser = data.groupBy(_._2)
//  println(s"Number of phones ${countByUser.count()}")
//  println(s"Number of phones with more then 5 checks ${countByUser.filter(_._2 > 5).count()}")
//  countByUser.collect().foreach(println)

  groupByUser.filter(_._2.size > 5).map(_._2.map(_._4.toInt)).map(mean(_)).filter(_ > -67).collect().foreach(println)

  def mean[T](item:Traversable[T])(implicit n:Numeric[T]) = {
    n.toDouble(item.sum) / item.size.toDouble
  }

//  val summary: MultivariateStatisticalSummary = Statistics.colStats(data.map(record => Vectors.dense(record._4)))
//  println(summary.mean)
//  println(summary.max)
//  println(summary.min)
//  println(summary.count)
//  println(summary.variance)
}
