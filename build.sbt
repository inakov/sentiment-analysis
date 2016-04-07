name := "sentiment-analysis"

version := "1.0"

scalaVersion := "2.11.7"

libraryDependencies += "org.scala-lang" % "scala-xml" % "2.11.0-M4"
libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.2.2"
libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.6.0"
libraryDependencies += "joda-time" % "joda-time" % "2.9.3"
libraryDependencies += "org.scalanlp" %% "breeze-viz" % "0.12"
libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.12"
libraryDependencies += "org.scalanlp" %% "breeze" % "0.12"