package pv021.network;

import pv021.data.Data;
import pv021.network.builder.LayerTemp;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {

    private final Data data;
    private final List<Layer> layers;
    private final double learningRate;
    private final Random random;

    public NeuralNetwork(Data data, List<LayerTemp> tempLayers, double learningRate, long seed) {
        this.data = data;
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.random = new Random(seed);
        initLayers(tempLayers);
    }

    private void initLayers(List<LayerTemp> tempLayers) {
        for (int i = 0; i < tempLayers.size(); i++) {
            LayerTemp layerTemp = tempLayers.get(i);
            LayerTemp layerTempNext = i == tempLayers.size() - 1 ? null : tempLayers.get(i + 1);
            layers.add(new Layer(
                    layerTemp.getSize(),
                    layerTempNext == null ? 0 : layerTempNext.getSize(),
                    layerTemp.getActivationFunction()));
        }
    }

    public void initializeWeights() {
        int n = data.getTrainVectors().get(0).size();

        for (Layer layer : layers) {
            if (!layer.isOutputLayer()) {
                for (int i = 0; i < layer.getSize(); i++) {
                    for (int j = 0; j < layer.getNextLayerSize() + 1; j++) {
                        layer.getWeights()[i][j] = random.nextGaussian(0, 2.0 / n);
                    }
                }
            }
        }
    }

    public void train() {

    }

    public void forward(int k) {

    }

    public Data getData() {
        return data;
    }

    public List<Layer> getLayers() {
        return layers;
    }
}
