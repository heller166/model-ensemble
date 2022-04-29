package ensemble

import org.apache.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

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

    var Array(trainingData, testData) = sc.textFile(args(0))
      .map(line => line.split(",").map(_.toDouble))
      // Split data into Array(Species, Length1, Length2, Length3, Height, Width) and Weight
      .map(arr => (Array(arr(0), arr(2), arr(3), arr(4), arr(5), arr(6)), arr(1)))
      .randomSplit(Array(0.7, 0.3))

    val modelsAndData: RDD[(LinearModel, Array[(Array[Double], Double)])] = sc.parallelize((0 to numModels)
      .map(_ => (new LinearModel(), trainingData.sample(withReplacement = true, 1.0).collect())))

    val trainedModels = modelsAndData.map{
        case (model, data) => model.train(
          data.map{case (samplePoints, _) => samplePoints},
          data.map{case (_, observation) => observation}
        )
    }

    //val testDataBroadcast = sc.broadcast(testData.collect())

    //val sumSquaredErr = trainedModels.map(
    //  model =>
    //    testDataBroadcast.value.map{
    //      case (samplePoints, observation) =>
    //        (model.predict(samplePoints), observation)
    //    }
    //)

    val testDataCollected = testData.collect()
    /*

    testData  id, data
    reduceByKey(

    model1  |   model2   |   model3

    trainedModel.map(model => {
      List predictions
      for samplePoint, observation in testData {
        predictions.append(model.predict())
      }
    }) // RDD[Array(Double)] size = number of models


    prediction1 model1  |  prediction1 model2 | ...

    => prediction1model1 + prediction1model2 ... / number of models
     */

    val sumSquaredErr = testDataCollected.map{
      case (samplePoints, observation) =>
        //val predictions = new ListBuffer[Double]
        val predictions = trainedModels.collect().map(model => model.predict(samplePoints))
        //trainedModels.foreach(model => println("here"))
        (predictions.sum / predictions.length, observation)
    }
    val sumSquaredErr2 = sumSquaredErr.map{
      case (prediction, observation) => scala.math.pow(observation - prediction, 2)
    }
    println(sumSquaredErr2.sum)
    //println(sumSquaredErr.toString)
    val meanObservations = testData.map{case(_, observation) => observation}.mean()
    println(meanObservations)
    val sumResiduals = testData.map{case(_, observation) => observation - meanObservations}.sum()
    println(sumResiduals)
    val r2Score = 1 - (sumResiduals / sumSquaredErr2.sum)

    println("R2 Score: " + r2Score)
  }
}
