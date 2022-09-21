package learn

import layer.Layer

class NetworkLearnData(val layerData: List<LayerLearnData>) {
    companion object {
        fun fromLayers(layers: List<Layer>): NetworkLearnData =
            NetworkLearnData(layers.map { LayerLearnData.fromLayer(it) })
    }
}
