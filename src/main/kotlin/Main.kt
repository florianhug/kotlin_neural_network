import datasource.DataPoints.toDataPoint
import datasource.mnist.MnistCompressedReader
import datasource.mnist.MnistDecompressedReader
import datasource.mnist.MnistEntry

fun main() {
    println("Hello Neural Network, please don't exterminate all hoomans")
    val trainingData: List<MnistEntry> = loadTrainingData()
    val dataPoints = trainingData.take(6000).map { it.toDataPoint(10) } //FIXME: Calculate number of labels

    val network = NeuralNetwork(listOf(784, 300, 10))
    network.learn(dataPoints, 0.05)
    trainingData.take(10)
//        .map { it.getImageData() }
        .forEach { it -> println("${it.label.toInt()} ${network.classify(it.getImageData())}") }
//        .forEach { println(it) }
}

fun loadTrainingData(): List<MnistEntry> {
    print("Loading training data ${MnistCompressedReader.trainImagesFileName} and ${MnistCompressedReader.trainLabelsFileName} ... ")
    val imagesResourceStream = Classification::class.java.getResourceAsStream(MnistCompressedReader.trainImagesFileName)
    val labelsResourceStream = Classification::class.java.getResourceAsStream(MnistCompressedReader.trainLabelsFileName)
    val reader = MnistCompressedReader(MnistDecompressedReader())
    val trainingData = mutableListOf<MnistEntry>()
    if (imagesResourceStream != null && labelsResourceStream != null) {
        reader.readCompressed(imagesResourceStream, labelsResourceStream) { trainingData.add(it) }
    } else {
        System.err.println("Unable to read resource files")
    }
    println(" completed. ${trainingData.size} entries loaded")
    return trainingData.toList()
}