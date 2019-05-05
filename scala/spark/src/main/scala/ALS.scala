import java.io.File

import org.apache.hadoop.fs.FileUtil
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object ALS {
  case class Event(userId: Int, publicId: Int, eventType: Int)

  def parseFiltered(str: String): Event = {
    val fields = str.split(" ")

    assert(fields.size == 2)

    Event(fields(0).toInt, fields(1).toInt.abs, if (fields(1).toInt < 0) -1 else 1)
  }

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("ALS")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    val filtered = spark
      .read
      .textFile("../../data/als_filtered")
      .map(parseFiltered)
      .toDF()

    val als = new ALS()
      .setMaxIter(5)
      .setRank(32)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("publicId")
      .setRatingCol("eventType")

    val model = als.fit(filtered)

    FileUtil.fullyDelete(new File("out/ALS_embeddings"))

    model
      .itemFactors
      .coalesce(1)
      .write
      .json("out/ALS_embeddings")

    spark.stop()
  }
}