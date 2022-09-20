package cost

interface Cost {
    fun cost(predictedOutputs: List<Double>, expectedOutputs: List<Double>): Double
    fun costDerivative(predictedOutput: Double, expectedOutput: Double): Double
}