package pv021.network;

import pv021.data.Data;
import pv021.network.builder.LayerTemp;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * General notes:
 * <p>
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

    public NeuralNetwork(Data data, List<LayerTemp> tempLayers, double learningRate, long seed, int steps, int batchSkip, double momentumAlpha, boolean debug) {
        this.data = data;
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.random = new Random(seed);
        this.steps = steps;
        this.batch = batchSkip;
        this.momentumAlpha = momentumAlpha;
        this.debug = debug;
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
    }

    public void initializeWeights() {
        int n = data.getTrainVectors().get(0).size();

        for (Layer layer : layers) {
            if (!layer.isOutputLayer()) {
                for (int j = 0; j < layer.getSize() + 1; j++) {
                    for (int r = 0; r < layer.getNextLayerSize(); r++) {
                        layer.getWeights()[r][j] = random.nextGaussian(0, 2.0 / n);
                    }
                }
            }
        }
    }

    public void trainBatch() {
        int p = data.getTrainVectors().size();
        int batchSize = Math.min(p, batch);
        System.err.println("P " + p);
        int t = 0;
        List<Integer> batches = IntStream.rangeClosed(0, p - 1).boxed().collect(Collectors.toList());
        while (t < steps) {
            if(debug) {
                printError();
            }
            Collections.shuffle(batches, random);
            for (int k : batches.subList(0, batchSize)) {
                forward(data.getTrainVectors().get(k));
                backpropagate(k);
                updateWeightsStep();
            }

            System.err.println(t+" | max = %f".formatted(Arrays.stream(layers.get(layers.size()-1).getPotentials()).max().orElse(0)));
            doStep();
            t++;
        }
        printError();
    }

    public <T extends Number> void forward(List<T> input) {
        Layer inputLayer = layers.get(0);
        for (int i = 0; i < inputLayer.getSize(); i++) {
            String str = input.get(i).toString();
            inputLayer.getOutputs()[i + 1] = Double.parseDouble(str);
        }

        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            IntStream.range(0, layer.getSize()).parallel().forEach(j -> {  //calculate potentials
                double potential = 0;

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    potential += previousLayer.getWeights()[j][i] * previousLayer.getOutputs()[i];
                }

                layer.getPotentials()[j] = potential;
            });

            double max = Arrays.stream(layer.getPotentials()).max().orElse(0);
            double sum = 0;

            for (int j = 0; j < layer.getSize(); j++) {   // sum of activation functions applied to potentials - optimisation
                sum += layer.getActivationFunction().apply(layer.getPotentials()[j], max);
            }

            for (int j = 0; j < layer.getSize(); j++) {   // calculate outputs
                layer.getOutputs()[j + 1] = layer.getActivationFunction().computeOutput(sum, layer.getPotentials()[j], max);
            }
        }
    }

    private void backpropagate(int k) {
        Layer outputLayer = layers.get(layers.size() - 1);
        for (int j = 0; j < outputLayer.getSize(); j++) {
            double y = outputLayer.getOutputs()[j + 1];
            double d = data.getTrainLabels().get(k).get(j);

            //outputLayer.getChainRuleTermWithOutput()[j] = -d / y + (1 - d) / (1 - y);
            outputLayer.getChainRuleTermWithOutput()[j] = y - d;
        }

        for (int l = layers.size() - 2; l >= 1; l--) {
            Layer nextLayer = layers.get(l + 1);
            Layer layer = layers.get(l);

            double max = Arrays.stream(layer.getPotentials()).max().orElse(0);
            double sum = IntStream.range(0, nextLayer.getSize()).mapToDouble(j -> nextLayer.getActivationFunction().apply(nextLayer.getPotentials()[j], max)).sum();

            // sum of activation functions applied to potentials - optimisation

            IntStream.range(0, layer.getSize()).parallel().forEach(j -> {
                double result = 0;

                for (int r = 0; r < nextLayer.getSize(); r++) {
                    double term1 = nextLayer.getChainRuleTermWithOutput()[r];
                    double term2 = nextLayer.getActivationFunction().computeDerivative(sum, nextLayer.getPotentials()[r], max);
                    double term3 = layer.getWeights()[r][j + 1];
                    result += term1 * term2 * term3;
                }

                layer.getChainRuleTermWithOutput()[j] = result;
            });
        }
    }

    private void updateWeightsStep() {
        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            double max = Arrays.stream(layer.getPotentials()).max().orElse(0);
            double sum = IntStream.range(0, layer.getSize()).mapToDouble(j -> layer.getActivationFunction().apply(layer.getPotentials()[j], max)).sum();

            IntStream.range(0, layer.getSize()).parallel().forEach(j -> {
                double term1 = layer.getChainRuleTermWithOutput()[j];
                double term2 = layer.getActivationFunction().computeDerivative(sum, layer.getPotentials()[j], max);
                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    double term3 = previousLayer.getOutputs()[i];;
                    double step = term1 * term2 * term3;
                    previousLayer.getWeightsStepAccumulator()[j][i] += step;
                }
            });
        }
    }

    private void doStep() {
        double alpha = 0.9;
        double ni = learningRate;
        double decay = 0.0000;
        double delta = 1e-8;
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
                    double currentRmsProp = alpha * rmsProp + (1 - alpha) * step * step;

                    double actualStep = - (ni / Math.sqrt(currentRmsProp + delta)) * step;
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


    private boolean printError() {
        double error = 0;
        Layer outputLayer = layers.get(layers.size() - 1);
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

                error += truth * Math.log(predicted);
            }
            total++;
            if (data.getTestLabels().get(k).get(result) == 1) {
                correct++;
            }
        }

        error *= -1.0/p;

        double pct = (correct * 100.0) / total;

        System.out.println(error + " / %f".formatted(pct));

        return error < 0.01;

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
