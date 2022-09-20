package cost

object MeanSquaredError : Cost {
    override fun cost(predictedOutputs: List<Double>, expectedOutputs: List<Double>): Double {
        var cost = 0.0
        for (i in predictedOutputs.indices) {
            val error = costDerivative(predictedOutputs[i], expectedOutputs[i])
            cost += error * error
        }
        return 0.5 * cost
    }

    override fun costDerivative(predictedOutput: Double, expectedOutput: Double): Double {
        return predictedOutput - expectedOutput
    }
}