package svm.training

/**
 * Created by inakov on 16-1-29.
 */
object Tokenizer {

  def tokenize(line: String, stopWords: Set[String]): Seq[String] = {
    line.split("""[^\p{L}\p{Nd}]+""")
      .map(_.toLowerCase)
      .filterNot(token => stopWords.contains(token))
      .filter(token => token.size >= 2).toSeq
  }

}
