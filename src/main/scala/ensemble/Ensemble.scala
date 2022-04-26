package ensemble

import org.apache.log4j.LogManager
import org.apache.spark
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

object Ensemble {

  def partition_data(num_machines: Int): Int = {
    scala.util.Random.nextInt(num_machines)
  }

  def main(args: Array[String]) {
    val logger: org.apache.log4j.Logger = LogManager.getRootLogger
    if (args.length != 3) {
      logger.error("Usage:\nensemble.Ensemble <train> <test> <output> <num_machines>")
      System.exit(1)
    }
    val conf = new SparkConf().setAppName("Ensemble")
    val sc = new SparkContext(conf)
    val num_machines = args(3).toInt

    val machineNumbers = List.range(0, num_machines)
    val machines = new ListBuffer[(Int, LinearModel)]()
    for ( machine <- 0 to num_machines) {
      machines += (machine, new LinearModel())
    }

    val machinesRdd = sc.parallelize(machines)

    val trainingCsv = sc.textFile(args(0))
    val testingCsv = sc.textFile(args(1))

    val trainedModels = trainingCsv.map(line => line.split(","))
      .map(line_values => (partition_data(num_machines), List(line_values.map(item => item.toDouble))))
      .reduceByKey((list1, list2) => list1 ++ list2)
      .fullOuterJoin(machinesRdd)
      .filter(values => values._2._1.isDefined && values._2._1.isDefined)
      .map {
        case (key, (dataset, model)) => (key, (dataset.get, model.get))
      }
      .map(values => (values._1, values._2._2.train(values._2._1)))

    val testModels = testingCsv.map(line => line.split(","))
      .map(values => values.map(a => a.toFloat))
      .map(dataPoint => {
        val predictions = new ListBuffer[Int]
        trainedModels.foreach(values => predictions ++ values._2.predict(dataPoint.toList))
        // Average
        //(dataPoint, predictions.sum / predictions.length)
        // Most occuring
        (dataPoint, predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1)
      })

    testModels.saveAsTextFile(args(2))
  }
}
