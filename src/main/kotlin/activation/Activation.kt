package activation

interface Activation {
    fun activate(inputs: List<Double>, index: Int): Double
    fun derivative(inputs: List<Double>, index: Int): Double
}
