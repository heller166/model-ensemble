package ensemble

import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Ensemble {
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
      .sample(withReplacement = false, 0.001)
      .map(line => line.split(",").map(_.toDouble))
      .map(arr => {
          val price = arr(8)
          val features = arr.patch(8,Nil,1).slice(0, 12)
          (features, price)
      })
      .randomSplit(Array(0.7, 0.3))

    val modelsAndData: RDD[(LinearModel, Array[(Array[Double], Double)])] = sc.parallelize((0 until numModels)
      .map(_ => (new LinearModel(), trainingData.sample(withReplacement = true, 1.0).collect())))

    val trainedModels = modelsAndData.map{
        case (model, data) => model.train(
          data.map{case (samplePoints, _) => samplePoints},
          data.map{case (_, observation) => observation}
        )
    }

    val testDataCollected = testData.collect()

    val sumSquaredError = testDataCollected.map{
      case (samplePoints, observation) =>
        val predictions = trainedModels.collect().map(model => model.predict(samplePoints))
        (predictions.sum / predictions.length, observation)
    }.map{
      case (prediction, observation) => scala.math.pow(observation - prediction, 2)
    }.sum
    val meanObservations = testData.map{case(_, observation) => observation}.mean()
    val sumSquaredTotal = testData.map{case(_, observation) => scala.math.pow(observation - meanObservations, 2)}.sum()
    val r2Score = 1 - (sumSquaredError / sumSquaredTotal)

    println("R2 Score: " + r2Score)
  }
}
