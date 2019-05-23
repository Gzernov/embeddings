import java.io.File

import org.apache.hadoop.fs.FileUtil
import org.apache.spark.SparkContext
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

    SparkContext.getOrCreate().setCheckpointDir("out/cdir")

    val topics = 64

    val lda = new LDA()
      .setK(topics)
      .setSeed(1994790107)
      .setCheckpointInterval(2)
      .setMaxIterations(50)

    val ldaTrained = lda.run(filtered)

    FileUtil.fullyDelete(new File("out/lda_embeddings"))

    val out = ldaTrained
      .topicsMatrix
      .transpose
      .toArray
      .grouped(ldaTrained.topicsMatrix.numCols)
      .toList
      .map(line => line.mkString(" "))

    SparkContext.getOrCreate().parallelize(out)
      .repartition(1)
      .saveAsTextFile("out/lda_embeddings")

    spark.stop()
  }
}