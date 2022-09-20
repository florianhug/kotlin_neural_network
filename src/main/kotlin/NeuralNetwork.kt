import activation.SigmoidActivation
import cost.Cost
import cost.MeanSquaredError
import layer.Layer

class NeuralNetwork(layerSizes: List<Int>) {
    private val layers = buildLayers(layerSizes)
    private val cost: Cost = MeanSquaredError

    fun classify(inputs: List<Double>): Classification {
        val outputs = calculateOutputs(inputs)
        val predictedClass = outputs.findMaxIndex()
        return Classification(predictedClass, outputs)
    }

    private fun calculateOutputs(inputs: List<Double>): List<Double> {
        var values = inputs.toList()
        for (layer in layers) {
            values = layer.calculateOutputs(values)
        }
        return values.toList()
    }

    companion object {
        private fun buildLayers(layerSizes: List<Int>): List<Layer> {
            val activation = SigmoidActivation
            val layerList = mutableListOf<Layer>()
            for (i in 0 until layerSizes.size - 1) {
                layerList.add(Layer(layerSizes[i], layerSizes[i + 1], activation))
            }
            return layerList.toList()
        }

        private fun List<Double>.findMaxIndex(): Int {
            var maxValue = 0.0
            var index = 0
            for (i in this.indices) {
                if (this[i] > maxValue) {
                    index = i
                    maxValue = this[0]
                }
            }
            return index
        }
    }
}