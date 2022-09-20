package datasource.mnist

import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte

/**
 * An entry of the MNIST data set. Instances of this class will be passed
 * to the consumer that is given to the [MnistCompressedReader] and
 * [MnistDecompressedReader] reading methods.
 */
class MnistEntry internal constructor(
    val index: Int,
    val label: Byte,
    private val numRows: Int,
    private val numCols: Int,
    private val imageData: ByteArray
) {

    fun getImageData(): List<Double> = imageData.map { it.toDouble() }

    /**
     * Creates a new buffered image from the image data that is stored
     * in this entry.
     *
     * @return The image
     */
    fun createImage(): BufferedImage {
        val image = BufferedImage(
            numCols,
            numRows, BufferedImage.TYPE_BYTE_GRAY
        )
        val dataBuffer = image.raster.dataBuffer
        val dataBufferByte = dataBuffer as DataBufferByte
        val data = dataBufferByte.data
        System.arraycopy(imageData, 0, data, 0, data.size)
        return image
    }

    override fun toString(): String {
        val indexString = String.format("%05d", index)
        return ("MnistEntry["
                + "index=" + indexString + ","
                + "label=" + label + "]")
    }
}