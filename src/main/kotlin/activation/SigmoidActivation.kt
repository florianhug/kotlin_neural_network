package activation

import kotlin.math.pow


object SigmoidActivation : Activation {
    override fun activate(inputs: List<Double>, index: Int): Double {
        return 1.0 / (1 + Math.E.pow(-inputs[index]))
    }

    override fun derivative(inputs: List<Double>, index: Int): Double {
        val a = activate(inputs, index)
        return a * (1 - a)
    }
}