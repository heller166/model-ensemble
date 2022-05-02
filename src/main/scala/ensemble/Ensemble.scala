package ensemble

import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer
import scala.util.Random

object Ensemble {

  def randomSample(data: Array[(Array[Float], Float)]): Array[(Array[Float], Float)] = {
    val sample = new ListBuffer[(Array[Float], Float)]
    for(_ <- data.indices) {
      Random.nextInt(data.length)
      sample.append(data(Random.nextInt(data.length)))
    }

    sample.toArray
  }

  def main(args: Array[String]): Unit = {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 2) {
      logger.error("Usage:\nensemble.Ensemble <input> <num_models>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Ensemble")
    val sc = new SparkContext(conf)
    val numModels = args(1).toInt

    val Array(trainingData, testData) = sc.textFile(args(0))
      .sample(withReplacement = false, 0.0025)
      .map(line => line.split(",").map(_.toFloat))
      .map(arr => {
          val price = arr(8)
          val features = arr.patch(8,Nil,1).slice(0, 12)
          (features, price)
      })
      .randomSplit(Array(0.7, 0.3))

    val trainingDataBroadcast = sc.broadcast(trainingData.collect())

    val models: RDD[LinearModel] = sc.parallelize(0 until numModels).map(_ => new LinearModel())

    val trainedModels = models.map(model => {
          val data = randomSample(trainingDataBroadcast.value)
          model.train(data.map{case (samplePoints, _) => samplePoints}, data.map{case (_, observation) => observation})
    })

    val testDataBroadcast = sc.broadcast(testData.collect())

    val sumSquaredError = trainedModels.flatMap(model => {
      val testData = testDataBroadcast.value
      val predictions = new ListBuffer[(Int, (Float, Float))]()
      for(i <- testData.indices) {
        predictions.append((i, (model.predict(testData(i)._1), testData(i)._2)))
      }
      predictions
    }).reduceByKey((a, b) => (a._1 + b._1, a._2))
      .map{
        case (_, (predictionSum, observation)) => (predictionSum / numModels, observation)
      }
      .map{
        case (prediction, observation) => scala.math.pow(observation - prediction, 2)
      }.sum()

    val meanObservations = testData.map{case(_, observation) => observation}.mean()
    val sumSquaredTotal = testData.map{case(_, observation) => scala.math.pow(observation - meanObservations, 2)}.sum()
    val r2Score = 1 - (sumSquaredError / sumSquaredTotal)

    println("R2 Score: " + r2Score)
  }
}
