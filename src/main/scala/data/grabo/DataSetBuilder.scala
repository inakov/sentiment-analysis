package data.grabo

import java.io.File

import com.github.tototoshi.csv.{CSVWriter, DefaultCSVFormat}
import org.apache.spark.{SparkConf, SparkContext}

import scala.xml.XML

/**
 * Created by inakov on 16-1-28.
 */
object DataSetBuilder extends App {

  def parseData() = {
    val graboXmlData = XML.loadFile("src/main/resources/dataset/grabo.xml")

    val businessData = (graboXmlData \ "business").map { business =>
      val cities = (business \ "cities").flatMap { city =>
        (city \ "city").map(_.text)
      }.toList

      val reviewData = (business \ "reviews").flatMap { reviewList =>
        (reviewList \ "review").map { review =>

          val rating = (review \ "rating").text match {
            case rating: String if rating.isEmpty => -1
            case rating: String => rating.toInt
          }
          Review((review \ "date").text, (review \ "user_id").text.toInt, rating, (review \ "message").text)
        }
      }.toList

      Business((business \ "id").text.toInt, (business \ "name").text, (business \ "category").text, cities, reviewData)
    }

    implicit object MyFormat extends DefaultCSVFormat {
      override val delimiter = '~'
    }

    val writer = CSVWriter.open(new File("src/main/resources/dataset/reviews.csv"))

    //  businessData.foreach{ business =>
    //    val line = List(business.id.toString, business.name.trim, business.category.trim, business.cities.mkString(" "))
    //    writer.writeRow(line)
    //  }

    for (business <- businessData) {
      val businessId = business.id.toString
      for (review <- business.reviews) {
        val msg = review.message.split('\n').map(_.trim.filter(_ >= ' ')).mkString
        val line = List(review.userId.toString, review.rating.toString, businessId, msg)
        writer.writeRow(line)
      }
    }

    writer.close()
    println("Done")
  }

  def splitDataset() = {
    val conf = new SparkConf().setAppName("Sentiment Analysis - DatasetBuilder")
      .setMaster("local[4]").set("spark.executor.memory", "1g")
    val sc = new SparkContext(conf)

    val reviewData = sc.textFile("src/main/resources/dataset/reviews.csv")
    val positiveReviews = reviewData.map(line => line.split("~")).filter(_.size > 3).collect {
      case review => (review(1).toInt, review(3))
    }.filter(_._1 > 3).randomSplit(Array(0.40, 0.20, 0.40))

    val negativeReviews = reviewData.map(line => line.split("~")).filter(_.size > 3).collect {
      case review => (review(1).toInt, review(3))
    }.filter(review => review._1 < 3 && review._1 > 0).randomSplit(Array(0.66, 0.33))


    implicit object MyFormat extends DefaultCSVFormat {
      override val delimiter = '~'
    }

    val validationSet = negativeReviews(1).union(positiveReviews(1)).collect()
    val valdationWriter = CSVWriter.open(new File("/home/inakov/GitHub/sentiment-analysis/src/main/resources/dataset/validation-set.csv"))
    validationSet.foreach(row => valdationWriter.writeRow(List(row._1, row._2)))

    val trainingSet = negativeReviews(0).union(positiveReviews(0)).collect()
    val trainingWriter = CSVWriter.open(new File("/home/inakov/GitHub/sentiment-analysis/src/main/resources/dataset/training-set.csv"))
    trainingSet.foreach(row => trainingWriter.writeRow(List(row._1, row._2)))

    valdationWriter.close()
    trainingWriter.close()
    println("Done")
  }

  //splitDataset()
}