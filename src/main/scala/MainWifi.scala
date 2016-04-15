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

  val rawData = sc.textFile("/home/inakov/Documents/trace-v3-10feed8ff65a-2016-03-26.csv")
  val data = rawData.map(line => line.split(",")).collect {
    case check => ((check(0).toDouble * 1000).toLong, check(2), check(3), check(4), check(5))
  }.filter(check => check._5 == "STA" && check._2 == "90e7c4ce3fe9").map(check => check.copy(_1 = check._1 - 1458943210000l)).cache()
//  data.take(100).foreach(println)
  val countByUser = data.map(check => (check._2, 1)).reduceByKey(_ + _).sortBy(-_._2)
  val groupByUser = data.groupBy(_._2)
//  println(s"Number of phones ${countByUser.count()}")
//  println(s"Number of phones with more then 5 checks ${countByUser.filter(_._2 > 5).count()}")
  countByUser.collect().foreach(println)

  val sessions = groupByUser.filter(_._2.size > 5).sortBy(-_._2.size).map(_._2.map(check => (check._1, check._4.toInt)))

  def mean[T](item:Traversable[T])(implicit n:Numeric[T]) = {
    n.toDouble(item.sum) / item.size.toDouble
  }

  def geometricMean[T](items:Traversable[T])(implicit n: Numeric[T]) = {
    val itemList = items.toList
    val len = itemList.length
    require (len > 0)
    require (! itemList.exists(n.toDouble(_) <= 0))

    len match {
      case 1 =>
        n.toDouble(itemList.head)

      case _ =>
        val recip = 1.0 / len.toDouble
        (1.0 /: itemList) ((a, b) => a * math.pow(n.toDouble(b), recip))
    }
  }

  def checkPoints(session: List[(Long, Int)], checks: List[List[(Long, Int)]]): List[List[(Long, Int)]] = {
    session match {
      case Nil => checks
      case list: List[(Long, Int)] => {
        val startTime = list.head._1
        val (checkPoint, tail) = list.partition(p => p._1 <= (startTime + 2000))
        checkPoints(tail, checkPoint::checks)
      }
    }
  }
//
////  val summary: MultivariateStatisticalSummary = Statistics.colStats(data.map(record => Vectors.dense(record._4)))
////  println(summary.mean)
////  println(summary.max)
////  println(summary.min)
////  println(summary.count)
////  println(summary.variance)
//
  val firstSession = sessions.take(1).last.toList
  val points = checkPoints(firstSession.map(ch => (ch._1, ch._2.toInt)), Nil).map(_.map(-_._2))
//  points.foreach(println)
//
  val means = points.map(mean(_))
  means.foreach(println)
  println()
  println()
  points.map(geometricMean(_)).foreach(println)

//  val scores = firstSession.map(_._2.toInt).toList
//  val count = scores.size
//  val mean = scores.sum / count
//  val devs = scores.map(score => (score - mean) * (score - mean))
//  val stddev = Math.sqrt(devs.sum / count)

//  println(s"Sum: $count")
//  println(s"Mean: $mean")
//  println(s"Devs: $devs")
//  println(s"Stddev: $stddev")
//
//  val points = data.map(_._4.toInt)
//
  import breeze.linalg._
//  import breeze.plot._
//
//  val f = Figure()
//  val p = f.subplot(0)

//  val x = new DenseVector(firstSession.map(_._1.toInt).toArray)
//  val y = new DenseVector(firstSession.map(_._2).toArray)
//  val x = new DenseVector(points.collect())
//  p += plot(x, x :^ 0)
//  p.xlabel = "time of the day in millis"
//  p.ylabel = "signal"
//  f.saveas("lines.png")


//  p += hist(points.collect(), 100)
//  p.title = "A normal distribution"
//  f.saveas("subplots.png")
}
