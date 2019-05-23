import java.io.File

import org.apache.hadoop.fs.FileUtil
import org.apache.spark.SparkContext
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object ALS {
  var MODE = 10

  case class Event(userId: Int, publicId: Int, eventType: Int)

  def parseFiltered(str: String): Event = {
    val fields = str.split(" ")

    var rating = 0

    if (MODE == 9){
      assert(fields.size == 2)

      rating = if (fields(1).toInt < 0) -1 else 1
    } else {
      assert(fields.size == 3)

      rating = fields(2).toInt
    }

    assert(rating != 0)

    Event(fields(0).toInt, fields(1).toInt.abs, rating)
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
      .textFile("../../data/als_filtered" + MODE)
      .map(parseFiltered)
      .toDF()

    SparkContext.getOrCreate().setCheckpointDir("out/cdir")

    val als = new ALS()
      .setMaxIter(200)
      .setRank(64)
      .setRegParam(0.01)
      .setSeed(1994790107)
      .setCheckpointInterval(2)
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