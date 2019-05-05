import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession

object LDA {
  var vocab_size: Int = -1

  def parseFiltered(x: (String, Long)): (Long, linalg.Vector) = {
    val arr = x._1.split(" ")

    val int_arr = arr.map(s => s.toInt)

    (x._2, Vectors.sparse(vocab_size, int_arr, arr.map(_ => 1.0)))
  }

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("LDA")
      .config("spark.master", "local")
      .getOrCreate()

    val raw = spark
      .read
      .textFile("../../data/lda_filtered")
      .rdd

    vocab_size = raw
      .collect()
      .map(s => s.split(" "))
      .map(s => s.toSet)
      .reduce((x, y) => x ++ y)
      .size

    val filtered = raw
      .zipWithIndex
      .map(parseFiltered)

    val topics = 32

    val lda = new LDA()
      .setK(topics)
      .setMaxIterations(5)

    val ldaTrained = lda.run(filtered)

    Files.write(
      Paths.get("out/lda_embeddings"),
      ldaTrained
        .topicsMatrix
        .toString(vocab_size, vocab_size)
        .getBytes(StandardCharsets.UTF_8)
    )

    spark.stop()
  }
}