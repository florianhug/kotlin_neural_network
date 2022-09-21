package datasource

import datasource.mnist.MnistEntry

object DataPoints {
    fun MnistEntry.toDataPoint(numberOfLabels: Int): DataPoint {
        return DataPoint(this.getImageData(), createOutput(this.label.toInt(), numberOfLabels), this.label.toInt())
    }

    private fun createOutput(label: Int, numberOfLabels: Int): List<Double> {
        return List(numberOfLabels) { index -> if (index == label) 1.0 else 0.0 }

    }
}