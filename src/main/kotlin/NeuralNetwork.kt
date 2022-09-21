import activation.SigmoidActivation
import cost.Cost
import cost.MeanSquaredError
import datasource.DataPoint
import layer.Layer
import learn.NetworkLearnData

class NeuralNetwork(layerSizes: List<Int>) {
    private val layers = buildLayers(layerSizes)
    private val cost: Cost = MeanSquaredError
    private var batchLearnData: List<NetworkLearnData>? = null

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

    fun learn(trainingData: List<DataPoint>, learnRate: Double, regularization: Double = 0.0, momentum: Double = 0.0) {
        print("Learning...")
        if (batchLearnData == null || batchLearnData?.size != trainingData.size) {
            batchLearnData = List(trainingData.size) { NetworkLearnData.fromLayers(layers) }
        }

        //TODO: Check if this can be executed concurrently
        for (trainingDataIndex in trainingData.indices) {
            updateGradients(trainingData[trainingDataIndex], batchLearnData!![trainingDataIndex])
        }
        for (layerIndex in layers.indices) {
            layers[layerIndex].applyGradient(learnRate / trainingData.size, momentum, regularization)
        }
        println(" done")

    }

    private fun updateGradients(dataPoint: DataPoint, learnData: NetworkLearnData) {
        var inputToNextLayer = dataPoint.inputs.toList()
        for (layerIndex in layers.indices) {
            inputToNextLayer = layers[layerIndex].calculateOutputs(inputToNextLayer, learnData.layerData[layerIndex])
        }
        val outputLayerIndex = layers.size - 1
        val outputLayer = layers[outputLayerIndex]
        val outputLearnData = learnData.layerData[outputLayerIndex]
        outputLayer.calculateOutputLayerNodeValues(outputLearnData, dataPoint.expectedOutputs, cost)
        outputLayer.updateGradients(outputLearnData)
        for (layerIndex in outputLayerIndex - 1 downTo 0) {
            val layerLearnData = learnData.layerData[layerIndex]
            val hiddenLayer = layers[layerIndex]
            hiddenLayer.calculateHiddenLayerNodeValues(
                layerLearnData,
                layers[layerIndex + 1],
                learnData.layerData[layerIndex + 1].nodeValues
            )
            hiddenLayer.updateGradients(layerLearnData)
        }
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
                    maxValue = this[i]
                }
            }
            return index
        }
    }
}