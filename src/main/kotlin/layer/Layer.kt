package layer

import activation.Activation
import cost.Cost
import layer.Layers.getWeight
import layer.Layers.randomlyInitializeWeights
import learn.LayerLearnData

class Layer(private val nodesIn: Int, val nodesOut: Int, private val activation: Activation) {
    private val weights: Array<Double> = randomlyInitializeWeights(nodesIn, nodesOut)
    private val biases: Array<Double> = Array(nodesOut) { 0.0 }

    // Cost gradient with respect to weights and with respect to biases
    private val costGradientW: MutableList<Double> = List(weights.size) { 0.0 }.toMutableList()
    private val costGradientB: MutableList<Double> = List(biases.size) { 0.0 }.toMutableList()

    private val weightVelocities: MutableList<Double> = List(weights.size) { 0.0 }.toMutableList()
    private val biasVelocities: MutableList<Double> = List(biases.size) { 0.0 }.toMutableList()

    fun calculateOutputs(inputs: List<Double>): List<Double> {
        val weightedInputs = List(inputs.size) { 0.0 }.toMutableList()
        for (nodeOut in 0 until nodesOut) {
            var weightedInput = biases[nodeOut]
            for (nodeIn in 0 until nodesIn) {
                weightedInput += inputs[nodeIn] * weights.getWeight(nodeIn, nodesIn, nodeOut)
            }
            weightedInputs[nodeOut] = weightedInput
        }
        val activations = List(nodesOut) { 0.0 }.toMutableList()
        for (outputNode in 0 until nodesOut) {
            activations[outputNode] = this.activation.activate(weightedInputs, outputNode)
        }
        return activations
    }

    fun calculateOutputs(inputs: List<Double>, learnData: LayerLearnData): List<Double> {
        learnData.inputs = inputs.toMutableList()
        for (nodeOut in 0 until nodesOut) {
            var weightedInput = biases[nodeOut]
            for (nodeIn in 0 until nodesIn) {
                weightedInput += inputs[nodeIn] * weights.getWeight(nodeIn, nodesIn, nodeOut)
            }
            learnData.weightedInputs[nodeOut] = weightedInput
        }

        for (activationIndex in 0 until learnData.activations.size) {
            learnData.activations[activationIndex] = activation.activate(learnData.weightedInputs, activationIndex)
        }
        return learnData.activations
    }

    fun calculateOutputLayerNodeValues(layerLearnData: LayerLearnData, expectedOutputs: List<Double>, cost: Cost) {
        for (nodeValueIndex in layerLearnData.nodeValues.indices) {
            val costDerivative =
                cost.costDerivative(layerLearnData.activations[nodeValueIndex], expectedOutputs[nodeValueIndex])
            val activationDerivative = activation.derivative(layerLearnData.weightedInputs, nodeValueIndex)
            layerLearnData.nodeValues[nodeValueIndex] = costDerivative * activationDerivative
        }
    }

    fun calculateHiddenLayerNodeValues(layerLearnData: LayerLearnData, oldLayer: Layer, oldNodeValues: List<Double>) {
        for (nodeIndex in 0 until nodesOut) {
            var newNodeValue = 0.0
            for (oldNodeIndex in oldNodeValues.indices) {
                val weightedInputDerivative = oldLayer.weights.getWeight(nodeIndex, oldLayer.nodesIn, oldNodeIndex)
                newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex]
            }
            newNodeValue *= activation.derivative(layerLearnData.weightedInputs, nodeIndex)
            layerLearnData.nodeValues[nodeIndex] = newNodeValue
        }
    }

    fun applyGradient(learnRate: Double, momentum: Double, regularization: Double) {
        updateWeights(learnRate, momentum, regularization)
        updateBiases(learnRate, momentum)
    }

    private fun updateBiases(learnRate: Double, momentum: Double) {
        for (biasIndex in biases.indices) {
            val velocity = biasVelocities[biasIndex] * momentum - costGradientB[biasIndex] * learnRate
            biasVelocities[biasIndex] = velocity
            biases[biasIndex] += velocity
            costGradientB[biasIndex] = 0.0
        }
    }

    private fun updateWeights(learnRate: Double, momentum: Double, regularization: Double) {
        val weightDecay = 1 - regularization * learnRate
        for (weightIndex in weights.indices) {
            val weight = weights[weightIndex]
            val velocity = weightVelocities[weightIndex] * momentum - costGradientW[weightIndex] * learnRate
            weightVelocities[weightIndex] = velocity
            weights[weightIndex] = weight * weightDecay + velocity
            costGradientW[weightIndex] = 0.0
        }
    }


    fun updateGradients(layerLearnData: LayerLearnData) {
        //Original code has a lock here (costGradientW)
        for (nodeOut in 0 until nodesOut) {
            val nodeValue = layerLearnData.nodeValues[nodeOut]
            for (nodeIn in 0 until nodesIn) {
                // Evaluate the partial derivative: cost / weight of current connection
                val derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue
                // The costGradientW array stores these partial derivatives for each weight.
                // Note: the derivative is being added to the array here because ultimately we want
                // to calculate the average gradient across all the data in the training batch
                costGradientW[getFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight
            }
        }
        //End of lock
        //Original code has a lock here (costGradientB)
        for (nodeOut in 0 until nodesOut) {
            val derivativeCostWrtBias = layerLearnData.nodeValues[nodeOut]
            costGradientB[nodeOut] += derivativeCostWrtBias
        }
        //End of lock
    }

    private fun getFlatWeightIndex(inputNeuronIndex: Int, outputNeuronIndex: Int): Int {
        return outputNeuronIndex * nodesIn + inputNeuronIndex
    }
}