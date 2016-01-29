package data.grabo

import java.io.File

import com.github.tototoshi.csv.{DefaultCSVFormat, CSVWriter}

import scala.xml.XML

/**
 * Created by inakov on 16-1-28.
 */
object DataSetBuilder extends App{
  val graboXmlData = XML.loadFile("/home/inakov/Downloads/sentiment-analysis/src/main/resources/grabo.xml")

  val businessData = (graboXmlData \ "business").map{ business =>
    val cities = (business \ "cities").flatMap{ city =>
      (city \ "city").map(_.text)
    }.toList

    val reviewData = (business \ "reviews").flatMap{ reviewList =>
      (reviewList \ "review").map{ review =>

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

  val writer = CSVWriter.open(new File("/home/inakov/Downloads/sentiment-analysis/reviews.csv"))

//  businessData.foreach{ business =>
//    val line = List(business.id.toString, business.name.trim, business.category.trim, business.cities.mkString(" "))
//    writer.writeRow(line)
//  }

  for(business <- businessData){
    val businessId = business.id.toString
    for(review <- business.reviews){
      val msg = review.message.split('\n').map(_.trim.filter(_ >= ' ')).mkString
      val line = List(review.userId.toString, review.rating.toString, businessId, msg)
      writer.writeRow(line)
    }
  }

  writer.close()
  println("Done")
}
