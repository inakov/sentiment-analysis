package data.grabo

/**
 * Created by inakov on 16-1-28.
 */

case class Review(date: String, userId: Int, rating: Int, message: String)

case class Business(id: Int, name: String, category: String, cities: List[String], reviews: List[Review])

case class Grabo(businesses: List[Business])


