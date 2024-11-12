package pv021.network;

import pv021.data.Data;
import pv021.network.builder.LayerTemp;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
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
                    for (int r = 0; r < layer.getNextLayerSize(); r++) {
                        layer.getWeights()[r][j] = random.nextGaussian(0, 2.0 / n);
                    }

                    layer.getBiases()[j] = random.nextGaussian(0, 2.0 / n);
                }
            }
        }
    }

    public void trainBatch() {
        for(int t = 0; t < 1000; t++) {

            // for each weight
            for (int l = 1; l < layers.size(); l++) {
                Layer previousLayer = layers.get(l - 1);
                Layer layer = layers.get(l);

                for(int j = 0; j < layer.getSize(); j++){
                    for(int i = 0; i < previousLayer.getSize(); i++) {
                        previousLayer.getWeights()[j][i] -= learningRate * computeErrorGradient(l, j, i);
                    }
                }
            }
        }
    }

    private double computeErrorGradient(int l, int j, int i) {
        int p = data.getTrainVectors().size();
        double result = 0.0;

        for(int k = 0; k < p; k++) {
            forward(data.getTrainLabels().get(k));
            backprop(k);
            result += computePartialGradient(l, j, i);
        }

        return result;
    }

    private double computePartialGradient(int l, int j, int i) {
        return layers.get(l).getChainRuleTermWithOutput()[j]
                * layers.get(l).getActivationFunction().applyDifferentiated(layers.get(l).getPotentials()[j])
                * layers.get(l - 1).getOutputs()[i];
    }

    private void backprop(int k) {
        Layer outputLayer = layers.get(layers.size() - 1);
        for (int j = 0; j < data.getLabelCount(); j++) {
            outputLayer.getChainRuleTermWithOutput()[j] = outputLayer.getOutputs()[j] - data.getTrainLabels().get(k).get(j);
        }

        for (int l = layers.size() - 2; l >= 0; l--) {
            Layer nextLayer = layers.get(l + 1);
            Layer layer = layers.get(l);

            for(int j = 0; j < layer.getSize(); j++) {
                double result = 0;

                for(int r = 0; r < nextLayer.getSize(); r++) {
                    double term1 = nextLayer.getChainRuleTermWithOutput()[r];
                    double term2 = nextLayer.getActivationFunction().applyDifferentiated(nextLayer.getPotentials()[r]);
                    double term3 = layer.getWeights()[r][j];
                    result = term1 * term2 * term3;
                }

                layer.getChainRuleTermWithOutput()[j] = result;
            }
        }
    }

    public void forward(List<Integer> input) {
        Layer inputLayer = layers.get(0);
        for(int i = 0; i < data.getLabelCount(); i++){
            inputLayer.getOutputs()[i] = input.get(i);
        }

        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for(int j = 0; j < layer.getSize(); j++){  //calculate potentials
                double potential = layer.getBiases()[j];

                for(int i = 0; i < previousLayer.getSize(); i++) {
                    potential += previousLayer.getWeights()[j][i] * previousLayer.getOutputs()[i];
                }

                layer.getPotentials()[j] = potential;
            }

            double sum = 0;

            for(int j = 0; j < layer.getSize(); j++) {   // sum of activation functions applied to potentials - optimisation
                sum += layer.getActivationFunction().apply(layer.getPotentials()[j]);
            }

            for(int j = 0; j < layer.getSize(); j++){   // calculate outputs
                layer.getOutputs()[j] = layer.getActivationFunction().computeOutput(sum, layer.getPotentials()[j]);
            }
        }

    }

    public Data getData() {
        return data;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void evaluate(String fileName) throws IOException {
        File csvOutputFile = new File(fileName);
        try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
            for(int k = 0; k < data.getTestLabels().size(); k++) {
                forward(data.getTestVectors().get(k));

                Layer outputLayer = layers.get(layers.size() - 1);
                double max = -Double.MAX_VALUE;
                int result = 0;
                for(int j = 0; j < outputLayer.getSize(); j++) {
                    double predicted = outputLayer.getOutputs()[j];
                    if(predicted  > max){
                        max = predicted;
                        result = j;
                    }
                    System.out.printf("Predicted: %f, expected: %d\n", predicted, data.getTestLabels().get(k).get(j));
                }
                System.out.printf("\n");

                pw.println(result);
            }
        }

    }
}
