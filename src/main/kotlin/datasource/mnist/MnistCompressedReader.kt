package datasource.mnist

import java.io.FileInputStream
import java.io.InputStream
import java.nio.file.Path
import java.util.function.Consumer
import java.util.zip.GZIPInputStream

class MnistCompressedReader(val decompressedReader: MnistDecompressedReader) {
    companion object {
        const val trainImagesFileName = "train-images-idx3-ubyte.gz"
        const val trainLabelsFileName = "train-labels-idx1-ubyte.gz"
    }

    fun readDecompressedTraining(inputDirectory: Path, consumer: Consumer<MnistEntry>) {
        val imagesFile = inputDirectory.resolve(trainImagesFileName)
        val labelsFile = inputDirectory.resolve(trainLabelsFileName)
        FileInputStream(imagesFile.toFile()).use { iis ->
            {
                FileInputStream(labelsFile.toFile()).use { lis ->
                    {
                        readCompressed(iis, lis, consumer)
                    }
                }
            }
        }
    }

    fun readCompressed(
        imagesInputStream: InputStream,
        labelInputStream: InputStream,
        consumer: Consumer<MnistEntry>
    ) {
        decompressedReader.readDecompressed(
            GZIPInputStream(imagesInputStream),
            GZIPInputStream(labelInputStream),
            consumer
        )
    }

}