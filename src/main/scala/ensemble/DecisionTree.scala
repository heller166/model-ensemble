package ensemble

import java.io.File
import org.apache.spark.sql.SparkSession.builder
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.lit

object DecisionTree {

  def main(args: Array[String]) {
    val spark = builder.master("local").appName("Decision Tree").getOrCreate()
    import org.apache.spark.ml.linalg._
    import spark.implicits._

    var training_directories = new File("input/training").listFiles()

    val trainingApricot = spark.read.format("image").load("input/training/Apricot")
      .select("image.data")
      .map(row => row.getAs[Array[Byte]]("data").map(byte => byte.toDouble))
      .map(row => Vectors.dense(row))
      .withColumn("label", lit(0))
    val trainingCocos = spark.read.format("image").load("input/training/Cocos")
      .select("image.data")
      .map(row => row.getAs[Array[Byte]]("data").map(byte => byte.toInt))
      .withColumn("label", lit(1))

    val trainingData = trainingApricot.union(trainingCocos)

    val testApricot = spark.read.format("image").load("input/training/Apricot")
      .select("image.data")
      .map(row => row.getAs[Array[Byte]]("data").map(byte => byte.toInt))
      .withColumn("label", lit(0))
    val testCocos = spark.read.format("image").load("input/training/Cocos")
      .select("image.data")
      .map(row => row.getAs[Array[Byte]]("data").map(byte => byte.toInt))
      .withColumn("label", lit(1))

    val testData = testApricot.union(testCocos)

    val assembler = new VectorAssembler()
      .setInputCols(Array("value"))
      .setOutputCol("value")

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("value")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(assembler, dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println(s"Learned regression tree model:\n ${treeModel.toDebugString}")
  }
}