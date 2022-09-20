import datasource.mnist.MnistCompressedReader
import datasource.mnist.MnistDecompressedReader
import datasource.mnist.MnistEntry

fun main() {
    println("Hello Neural Network, please don't exterminate all hoomans")
    val trainingData = loadTrainingData()

    val network = NeuralNetwork(listOf(784,300,10))
    val classification = network.classify(trainingData.first().getImageData())
    println(classification)
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