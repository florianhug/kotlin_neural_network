package layer

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log
import kotlin.math.sqrt
import kotlin.random.Random

internal object Layers {
    private val random: Random = Random(6314)

    fun randomlyInitializeWeights(numberOfWeights: Int, nodesIn: Int): Array<Double> = Array(numberOfWeights) {
        randomInNormalDistribution() / sqrt(nodesIn.toDouble())
    }

    private fun randomInNormalDistribution(mean: Double = 0.0, standardDeviation: Double = 1.0): Double {
        val x0 = 1 - random.nextDouble()
        val x1 = 1 - random.nextDouble()
        val y0 = normalDistribution(x0, x1)
        return y0 * standardDeviation + mean
    }

    private fun normalDistribution(x: Double, y: Double) = sqrt(-2.0 * log(x, Math.E)) * cos(2.0 * PI * y)
}