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
    private final int steps;

    public NeuralNetwork(Data data, List<LayerTemp> tempLayers, double learningRate, long seed, int steps) {
        this.data = data;
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.random = new Random(seed);
        this.steps=steps;
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
                for (int j = 0; j < layer.getSize() + 1; j++) {
                    for (int r = 0; r < layer.getNextLayerSize(); r++) {
                        layer.getWeights2()[r][j] = random.nextGaussian(0, 2.0 / n);
                    }
                }
            }
        }
    }

    public void trainBatch() {
        int p = data.getTrainVectors().size();
        for (int t = 0; t < steps; t++) {

            int k = t % p;
            //for (int k = 0; k < p; k++) {
                forward(data.getTrainVectors().get(k));
                backpropagate(k);
                updateWeightsStep();
           // }

            doStep();
            if(printError()){
                break;
            }
        }
    }

    private void doStep() {
        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {
                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    previousLayer.getWeights2()[j][i] += -learningRate * previousLayer.getWeightsStepAccumulator2()[j][i];
                    previousLayer.getWeightsStepAccumulator2()[j][i] = 0;
                }
            }
        }
    }

    private void updateWeightsStep() {
        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {
                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    double step = layer.getChainRuleTermWithOutput2()[j]
                            * layer.getActivationFunction().applyDifferentiated(layer.getPotentials()[j])
                            * previousLayer.getOutputs2()[i];
                    previousLayer.getWeightsStepAccumulator2()[j][i] += step;
                }
            }
        }
    }

    private void backpropagate(int k) {
        Layer outputLayer = layers.get(layers.size() - 1);
        for (int j = 0; j < outputLayer.getSize(); j++) {
            double y = outputLayer.getOutputs2()[j + 1];
            double d = data.getTrainLabels().get(k).get(j);
            outputLayer.getChainRuleTermWithOutput2()[j] = -d * Math.log(y) - (1 - d) * Math.log(1 - y);;//y - d;
        }

        for (int l = layers.size() - 2; l >= 0; l--) {
            Layer nextLayer = layers.get(l + 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize() + 1; j++) {
                double result = 0;

                for (int r = 0; r < nextLayer.getSize(); r++) {
                    double term1 = nextLayer.getChainRuleTermWithOutput2()[r];
                    double term2 = nextLayer.getActivationFunction().applyDifferentiated(nextLayer.getPotentials()[r]);
                    double term3 = layer.getWeights2()[r][j];
                    result = term1 * term2 * term3;
                }

                layer.getChainRuleTermWithOutput2()[j] = result;
            }
        }
    }

    public void forward(List<Integer> input) {
        Layer inputLayer = layers.get(0);
        for (int i = 0; i < data.getLabelCount(); i++) {
            inputLayer.getOutputs2()[i + 1] = input.get(i);
        }

        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {  //calculate potentials
                double potential = 0;

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    potential += previousLayer.getWeights2()[j][i] * previousLayer.getOutputs2()[i];
                }

                layer.getPotentials()[j] = potential;
            }

            double sum = 0;

            for (int j = 0; j < layer.getSize(); j++) {   // sum of activation functions applied to potentials - optimisation
                sum += layer.getActivationFunction().apply(layer.getPotentials()[j]);
            }

            for (int j = 0; j < layer.getSize(); j++) {   // calculate outputs
                layer.getOutputs2()[j + 1] = layer.getActivationFunction().computeOutput(sum, layer.getPotentials()[j]);
            }
        }

    }

    public Data getData() {
        return data;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    private boolean printError() {
        double error = 0;
        Layer outputLayer = layers.get(layers.size() - 1);
        for(int k = 0; k < data.getTestVectors().size(); k++){
            forward(data.getTestVectors().get(k));
            for(int j = 0; j < outputLayer.getSize(); j++){
                double predicted = outputLayer.getOutputs2()[j + 1];
                double truth = data.getTestLabels().get(k).get(j);
                error += ((predicted - truth) * (predicted - truth)) / 2.0;
            }
        }
        System.out.println(error);

        return error < 0.15;

    }

    public void evaluate(String fileName) throws IOException {
        System.out.println("==============");
        File csvOutputFile = new File(fileName);
        try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
            for (int k = 0; k < data.getTestVectors().size(); k++) {
                forward(data.getTestVectors().get(k));

                Layer outputLayer = layers.get(layers.size() - 1);
                double max = -Double.MAX_VALUE;
                int result = 0;
                for (int j = 0; j < outputLayer.getSize(); j++) {
                    double predicted = outputLayer.getOutputs2()[j + 1];
                    double truth = data.getTestLabels().get(k).get(j);
                    if (predicted > max) {
                        max = predicted;
                        result = j;
                    }
                    System.out.printf("Predicted: %f, expected: %f\n", predicted, truth);
                }
                System.out.printf("\n");

                pw.println(result);
            }
        }

    }
}
