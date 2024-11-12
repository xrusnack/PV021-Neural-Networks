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
                for (int j = 0; j < layer.getSize(); j++) {
                    for (int k = 0; k < layer.getNextLayerSize(); k++) {
                        layer.getWeights()[j][k] = random.nextGaussian(0, 2.0 / n);
                    }

                    layer.getBiases()[j] = random.nextGaussian(0, 2.0 / n);
                }
            }
        }
    }

    public void train() {

    }

    public void forward(int k) {
        Layer inputLayer = layers.get(0);
        for(int i = 0; i < data.getLabelCount(); i++){
            inputLayer.getOutputs()[i] = data.getTrainLabels().get(k).get(i);
        }

        for (int l = 1; l < layers.size(); l++) {

            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for(int j = 0; j < layer.getSize(); j++){
                double potential = layer.getBiases()[j];

                for(int i = 0; i < previousLayer.getSize(); i++) {
                    potential += previousLayer.getWeights()[j][i] * previousLayer.getOutputs()[i];
                }

                layer.getPotentials()[j] = potential;
                layer.getOutputs()[j] = layer.getActivationFunction().apply(potential);
            }



        }
    }

    public Data getData() {
        return data;
    }

    public List<Layer> getLayers() {
        return layers;
    }
}
