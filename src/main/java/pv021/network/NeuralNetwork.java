package pv021.network;

import pv021.data.Data;
import pv021.network.builder.LayerTemp;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final Data data;
    private final List<Layer> layers;

    public NeuralNetwork(Data data, List<LayerTemp> tempLayers) {
        this.data = data;
        this.layers = new ArrayList<>();
        initLayers(tempLayers);
    }

    private void initLayers(List<LayerTemp> tempLayers) {
        for (int i = 0; i < tempLayers.size(); i++) {
            LayerTemp layerTemp = tempLayers.get(i);
            LayerTemp layerTempNext = i == tempLayers.size() - 1 ? null : tempLayers.get(i + 1);
            layers.add(new Layer(layerTemp.getSize(), layerTempNext == null ? 0 : layerTempNext.getSize(), layerTemp.getActivationFunction()));
        }
    }

    public Data getData() {
        return data;
    }

    public List<Layer> getLayers() {
        return layers;
    }
}
