package layer

import activation.Activation
import layer.Layers.randomlyInitializeWeights

class Layer(private val nodesIn: Int, private val nodesOut: Int, private val activation: Activation) {
    private val weights: Array<Double> = randomlyInitializeWeights(nodesIn * nodesOut, nodesIn)
    private val biases: Array<Double> = Array(nodesOut) { 0.0 }

    fun calculateOutputs(inputs: List<Double>): List<Double> {
        val weightedInputs = List(inputs.size) { 0.0 }.toMutableList()
        for (nodeOut in 0 until nodesOut) {
            var weightedInput = biases[nodeOut]
            for (nodeIn in 0 until nodesIn) {
                weightedInput += inputs[nodeIn] * weights.getWeight(nodeIn, nodeOut)
            }
            weightedInputs[nodeOut] = weightedInput
        }
        val activations = List(nodesOut) { 0.0 }.toMutableList()
        for (outputNode in 0 until nodesOut) {
            activations[outputNode] = this.activation.activate(inputs, outputNode)
        }
        return activations
    }

    private fun Array<Double>.getWeight(nodeIn: Int, nodeOut: Int): Double {
        val index = nodeOut * nodesIn + nodeIn
        return this[index]
    }
}