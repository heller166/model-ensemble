package ensemble

import org.ejml.data.FMatrixRMaj
import org.ejml.dense.row.CommonOps_FDRM.dot
import org.ejml.dense.row.factory.LinearSolverFactory_FDRM
import org.ejml.dense.row.linsol.AdjustableLinearSolver_FDRM

class LinearModel(var weights: FMatrixRMaj = null) extends Serializable {
  def train(samplePoints: Array[Array[Float]], observations: Array[Float]): LinearModel = {
    this.weights = new FMatrixRMaj(samplePoints(0).length, 1)
    val A: FMatrixRMaj = new FMatrixRMaj(samplePoints)
    val y: FMatrixRMaj = new FMatrixRMaj(observations)

    val solver: AdjustableLinearSolver_FDRM = LinearSolverFactory_FDRM.adjustable()

    if(!solver.setA(A)) {
      throw new RuntimeException("Solver failed")
    }

    solver.solve(y, weights)
    this
  }

  def predict(row: Array[Float]): Float = {
    dot(weights, new FMatrixRMaj(row))
  }
}

