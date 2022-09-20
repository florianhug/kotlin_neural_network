package datasource.mnist

import java.io.DataInputStream
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.nio.file.Path
import java.util.function.Consumer

class MnistDecompressedReader {
    private companion object {
        const val trainImagesFileName = "train-images.idx3-ubyte"
        const val trainLabelsFileName = "train-labels.idx1-ubyte"

        private fun read(inputStream: InputStream, data: ByteArray) {
            var offset = 0
            while (true) {
                val read = inputStream.read(data, offset, data.size - offset)
                if (read < 0) {
                    break
                }
                offset += read
                if (offset == data.size) {
                    return
                }
            }
            throw IOException("Tried to read ${data.size} bytes, but only found $offset")
        }
    }

    fun readDecompressedTraining(inputDirectory: Path, consumer: Consumer<MnistEntry>) {
        val imagesFile = inputDirectory.resolve(trainImagesFileName)
        val labelsFile = inputDirectory.resolve(trainLabelsFileName)
        FileInputStream(imagesFile.toFile()).use { iis ->
            {
                FileInputStream(labelsFile.toFile()).use { lis ->
                    {
                        readDecompressed(iis, lis, consumer)
                    }
                }
            }
        }
    }

    fun readDecompressed(
        imagesInputStream: InputStream,
        labelsInputStream: InputStream,
        consumer: Consumer<MnistEntry>
    ) {
        val imagesDataInputStream = DataInputStream(imagesInputStream)
        val labelsDataInputStream = DataInputStream(labelsInputStream)
        val imagesMagic = imagesDataInputStream.readInt()
        if (imagesMagic != 0x803) {
            throw IOException("Expected magic header of 0x803 for images, but found $imagesMagic")
        }

        val labelMagic = labelsDataInputStream.readInt()
        if (labelMagic != 0x801) {
            throw IOException("Expected magic header of 0x801 for images, but found $labelMagic")
        }

        val numberOfImages = imagesDataInputStream.readInt()
        val numberOfLabels = labelsDataInputStream.readInt()
        if (numberOfImages != numberOfLabels) {
            throw IOException("Found $numberOfImages  images but $numberOfLabels labels")
        }

        val numRows = imagesDataInputStream.readInt()
        val numCols = imagesDataInputStream.readInt()

        for (i in 0 until numberOfImages) {
            val label = labelsDataInputStream.readByte()
            val imageData = ByteArray(numRows * numCols)
            read(imagesDataInputStream, imageData)
            val mnistEntry = MnistEntry(i, label, numRows, numCols, imageData)
            consumer.accept(mnistEntry)
        }
    }
}