package pv021.network;

import pv021.data.Data;
import pv021.function.error.CrossEntropy;
import pv021.function.error.ErrorFunction;
import pv021.network.builder.LayerTemp;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The class representing the neural network.
 * <p>
 * General notes:
 * j denotes the index of the neuron in the current l-th layer
 * i denotes the index of the neuron in the (l-1)-th layer
 * r denotes the index of the neuron in the (l+1)-th layer
 * w_ji denotes the weight of the connection from neuron i (in the (l-1)-th layer) to neuron j (in the l-th layer)
 */

public class NeuralNetwork {

    private final Data data;
    private final List<Layer> layers;
    private final double learningRate;
    private final Random random;
    private final int steps;
    private final int batch;
    private final double momentumAlpha;
    private final boolean debug;
    private final double rmsAlpha;
    private final double decay;

    public NeuralNetwork(Data data, List<LayerTemp> tempLayers,
                         double learningRate, long seed, int steps, int batchSkip, double momentumAlpha, boolean debug,
                         double rmsAlpha, double decay) {
        this.data = data;
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.random = new Random(seed);
        this.steps = steps;
        this.batch = batchSkip;
        this.momentumAlpha = momentumAlpha;
        this.debug = debug;
        this.rmsAlpha = rmsAlpha;
        this.decay = decay;
        initLayers(tempLayers);
    }

    private void initLayers(List<LayerTemp> tempLayers) {
        for (int i = 0; i < tempLayers.size(); i++) {
            LayerTemp layerTemp = tempLayers.get(i);
            LayerTemp layerTempNext = i == tempLayers.size() - 1 ? null : tempLayers.get(i + 1);
            layers.add(new Layer(
                    layerTemp.getSize(),
                    layerTempNext == null ? 0 : layerTempNext.getSize(),
                    layerTemp.getActivationFunction(), i == 0));
        }
        initializeWeights();  // we use the Normal He-initialization
    }

    public void initializeWeights() {
        int n = data.getTrainVectors().get(0).size();

        for (Layer layer : layers) {
            if (!layer.isOutputLayer()) {
                for (int j = 0; j < layer.getSize() + 1; j++) {
                    for (int r = 0; r < layer.getNextLayerSize(); r++) {
                        layer.getWeights()[r][j] = j == 0 ? 0 : random.nextGaussian(0, 2.0 / n);
                    }
                }
            }
        }
    }

    public void stochasticGradientDescent() {
        int p = data.getTrainVectors().size();  // number of training examples
        int batchSize = Math.min(p, batch);
        List<Integer> batches = IntStream.rangeClosed(0, p - 1).boxed().collect(Collectors.toList()); // choose a minibatch

        for (int t = 0; t < steps; t++) {
            if (debug && t % 1000 == 0) {
                printError();
            }
            Collections.shuffle(batches, random);
            for (int k : batches.subList(0, batchSize)) {
                forward(data.getTrainVectors().get(k));
                backpropagate(k, new CrossEntropy());
                computeGradient();
            }
            updateWeights();
            //System.err.println(t + " | max = %f min = %f".formatted(
            //        Arrays.stream(layers.get(layers.size() - 1).getPotentials()).max().orElse(0),
            //        Arrays.stream(layers.get(layers.size() - 1).getPotentials()).min().orElse(0)));
        }
        printError();
    }

    public <T extends Number> void forward(List<T> input) {
        Layer inputLayer = layers.get(0);

        for (int i = 0; i < inputLayer.getSize(); i++) {  // start the forward pass by evaluating the input neurons
            String str = input.get(i).toString();
            inputLayer.getOutputs()[i + 1] = Double.parseDouble(str);
        }

        for (int l = 1; l < layers.size(); l++) {  // forward pass in the hidden layers
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            IntStream.range(0, layer.getSize()).parallel().forEach(j -> {  // calculate potentials
                double potential = 0;

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    potential += previousLayer.getWeights()[j][i] * previousLayer.getOutputs()[i];
                }
                layer.getPotentials()[j] = potential;
            });

            double max = Arrays.stream(layer.getPotentials()).max().orElse(0);

            // calculate sum of activation functions applied to potentials (for the output layer with softmax) - optimisation
            double sum = 0;
            for (int j = 0; j < layer.getSize(); j++) {
                sum += layer.getActivationFunction().apply(layer.getPotentials()[j], max);
            }

            // calculate outputs for each neuron in the layer
            for (int j = 0; j < layer.getSize(); j++) {
                layer.getOutputs()[j + 1] = layer.getActivationFunction()
                        .computeOutput(sum, layer.getPotentials()[j], max);
            }
        }
    }

    private void backpropagate(int k, ErrorFunction errorFunction) {
        Layer outputLayer = layers.get(layers.size() - 1);
        for (int j = 0; j < outputLayer.getSize(); j++) {  // start with the output layer
            double y = outputLayer.getOutputs()[j + 1];
            double d = data.getTrainLabels().get(k).get(j);

            // compute the partial derivative of the errorFunction with respect to the outputs
            outputLayer.getChainRuleTermWithOutput()[j] = errorFunction.calculatePartialDerivative(y, d);
        }

        // compute the partial derivative of the errorFunction with respect to the outputs in the hidden layers
        for (int l = layers.size() - 2; l >= 1; l--) {
            Layer nextLayer = layers.get(l + 1);
            Layer layer = layers.get(l);

            double max = Arrays.stream(nextLayer.getPotentials()).max().orElse(0);
            double sum = IntStream.range(0, nextLayer.getSize()).mapToDouble(j ->
                    nextLayer.getActivationFunction().apply(nextLayer.getPotentials()[j], max)).sum();

            // sum of activation functions applied to potentials - optimisation
            IntStream.range(0, layer.getSize()).parallel().forEach(j -> {
                double result = 0;

                for (int r = 0; r < nextLayer.getSize(); r++) {
                    result += nextLayer.getChainRuleTermWithOutput()[r]
                            * nextLayer.getActivationFunction().computeDerivative(sum, nextLayer.getPotentials()[r], max)
                            * layer.getWeights()[r][j + 1];
                }
                layer.getChainRuleTermWithOutput()[j] = result;
            });
        }
    }

    private void computeGradient() {
        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            double max = Arrays.stream(layer.getPotentials()).max().orElse(0);
            double sum = IntStream.range(0, layer.getSize()).mapToDouble(j ->
                    layer.getActivationFunction().apply(layer.getPotentials()[j], max)).sum();

            IntStream.range(0, layer.getSize()).parallel().forEach(j -> {
                double term1 = layer.getChainRuleTermWithOutput()[j];
                double term2 = layer.getActivationFunction().computeDerivative(sum, layer.getPotentials()[j], max);

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    double term3 = previousLayer.getOutputs()[i];

                    double step = term1 * term2 * term3;
                    previousLayer.getWeightsStepAccumulator()[j][i] += step;  // accumulate the partial derivatives
                }
            });
        }
    }

    private void updateWeights() {  // update the weights + optimize with momentum and RMSProp
        double delta = 1e-8;  // smoothing term to avoid division by zero

        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {
                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    // the big sum
                    double step = previousLayer.getWeightsStepAccumulator()[j][i];

                    // r_ji^(t - 1)
                    double rmsProp = previousLayer.getRmsprop()[j][i];

                    // r_ji^(t)
                    double currentRmsProp = rmsAlpha * rmsProp + (1 - rmsAlpha) * step * step;

                    double actualStep = -(learningRate / Math.sqrt(currentRmsProp + delta)) * step;
                    double previousStep = previousLayer.getMomentum()[j][i];

                    double momentumBalancedStep = actualStep * (1 - momentumAlpha) + momentumAlpha * previousStep;

                    previousLayer.getWeights()[j][i] *= (1 - decay);
                    previousLayer.getWeights()[j][i] += momentumBalancedStep;
                    previousLayer.getWeightsStepAccumulator()[j][i] = 0;
                    previousLayer.getRmsprop()[j][i] = currentRmsProp;
                    previousLayer.getMomentum()[j][i] = momentumBalancedStep;
                }
            }
        }
    }


    private void printError() {
        Layer outputLayer = layers.get(layers.size() - 1);

        double errorTest = 0;
        int total = 0;
        int correct = 0;

        int p = data.getTestVectors().size();
        for (int k = 0; k < p; k++) {
            forward(data.getTestVectors().get(k));
            double max = -Double.MAX_VALUE;
            int result = 0;
            for (int j = 0; j < outputLayer.getSize(); j++) {
                double predicted = outputLayer.getOutputs()[j + 1];
                int truth = data.getTestLabels().get(k).get(j);
                if (predicted > max) {
                    max = predicted;
                    result = j;
                }

                errorTest -= (truth * Math.log(predicted) + (1 - truth) * Math.log(1 - predicted)) / p;
            }
            total++;
            if (data.getTestLabels().get(k).get(result) == 1) {
                correct++;
            }
        }

        double accuracyTest = (correct * 100.0) / total;

        double errorTrain = 0;
        total = 0;
        correct = 0;

        p = data.getTrainVectors().size();
        for (int k = 0; k < p; k++) {
            forward(data.getTrainVectors().get(k));
            double max = -Double.MAX_VALUE;
            int result = 0;
            for (int j = 0; j < outputLayer.getSize(); j++) {
                double predicted = outputLayer.getOutputs()[j + 1];
                int truth = data.getTrainLabels().get(k).get(j);
                if (predicted > max) {
                    max = predicted;
                    result = j;
                }

                errorTrain -= (truth * Math.log(predicted) + (1 - truth) * Math.log(1 - predicted)) / p;
            }
            total++;
            if (data.getTrainLabels().get(k).get(result) == 1) {
                correct++;
            }
        }

        double accuracyTrain = (correct * 100.0) / total;
        System.out.printf("test: [loss: %.6f accuracy: %.2f%%] train: [loss: %.6f accuracy: %.2f%%] overfit: %.6f / %.2f%%%n", errorTest, accuracyTest, errorTrain, accuracyTrain, -errorTrain + errorTest, accuracyTrain - accuracyTest);
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
                    double predicted = outputLayer.getOutputs()[j + 1];
                    double truth = data.getTestLabels().get(k).get(j);
                    if (predicted > max) {
                        max = predicted;
                        result = j;
                    }
                    // System.out.printf("Predicted: %f, expected: %f\n", predicted, truth);
                }
                // System.out.printf("\n");

                pw.println(result);
            }
        }

    }

    public void drawDistribution() {
        int w = 200;
        int h = 200;

        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);

        Layer outputLayer = layers.get(layers.size() - 1);
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                List<Double> input = List.of(x / (w - 1.0), y / (h - 1.0));
                forward(input);

                double max = -Double.MAX_VALUE;
                int result = 0;
                for (int j = 0; j < outputLayer.getSize(); j++) {
                    double predicted = outputLayer.getOutputs()[j + 1];
                    if (predicted > max) {
                        max = predicted;
                        result = j;
                    }
                }

                double pct = result / (data.getLabelCount() - 1.0);
                img.setRGB(x, y, new Color(0, (int) (255 * pct), 0).getRGB());
            }
        }

        try {
            ImageIO.write(img, "PNG", new File("distribution.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
