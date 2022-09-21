package learn

import layer.Layer

//TODO: make this entirely immutable
data class LayerLearnData(
    var inputs: MutableList<Double>,
    val weightedInputs: MutableList<Double>,
    val activations: MutableList<Double>,
    val nodeValues: MutableList<Double>
) {
    companion object {
        internal fun fromLayer(layer: Layer): LayerLearnData {
            return LayerLearnData(
                List(layer.nodesOut) { 0.0 }.toMutableList(),
                List(layer.nodesOut) { 0.0 }.toMutableList(),
                List(layer.nodesOut) { 0.0 }.toMutableList(),
                List(layer.nodesOut) { 0.0 }.toMutableList()
            )
        }
    }
}
