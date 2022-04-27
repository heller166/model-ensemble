package ensemble

import org.apache.spark.sql.SparkSession.builder
import org.ejml.data.DMatrixRMaj
import org.ejml.dense.row.CommonOps_DDRM.dot
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM
import org.ejml.dense.row.linsol.AdjustableLinearSolver_DDRM

class LinearModel(var weights: DMatrixRMaj = null) {
  def train(samplePoints: Array[Array[Double]], observations: Array[Double]): LinearModel = {
    this.weights = new DMatrixRMaj(samplePoints(0).length, 1)
    val A: DMatrixRMaj = new DMatrixRMaj(samplePoints)
    val y: DMatrixRMaj = new DMatrixRMaj(observations)

    val solver: AdjustableLinearSolver_DDRM = LinearSolverFactory_DDRM.adjustable()

    if(!solver.setA(A)) {
      throw new RuntimeException("Solver failed")
    }

    solver.solve(y, weights)
    this
  }

  def predict(row: Array[Double]): Double = {
    dot(weights, new DMatrixRMaj(row))
  }
}

object LinearModel {
  def main(args: Array[String]): Unit = {
    val spark = builder.master("local").appName("Ensemble").getOrCreate()

    val data = spark.read.option("header", "true")
      .schema("Species DOUBLE, Weight DOUBLE, Length1 DOUBLE, Length2 DOUBLE, Length3 DOUBLE, Height DOUBLE, Width DOUBLE")
      .csv(args(0)).toDF()
    val Array(train, test) = data.randomSplit(Array(0.7, 0.3))

    val model = new LinearModel()
    model.train(
      train.select("Species", "Length1", "Length2", "Length3", "Height", "Width").collect().map(_.toSeq.toArray.map(_.toString.toDouble)),
      train.select("Weight").collect().map(_.getDouble(0))
    )

    val samples = test.select("Species", "Length1", "Length2", "Length3", "Height", "Width")
      .first().toSeq.toArray.map(_.toString.toDouble)
    val observation = test.select("Weight").first().getDouble(0)

    println("Real Weight:", observation)
    println("Predicted Weight:", model.predict(samples))
  }
}

