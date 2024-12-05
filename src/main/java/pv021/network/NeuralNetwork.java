package pv021.network;

import pv021.data.Data;
import pv021.function.error.CrossEntropy;
import pv021.function.error.ErrorFunction;
import pv021.network.builder.LayerTemplate;

import java.io.File;
import java.io.PrintWriter;
import java.util.List;
import java.util.Random;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
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
    private final double rmsAlpha;
    private final ErrorFunction errorFunction = new CrossEntropy();

    public static int threads = Runtime.getRuntime().availableProcessors();
    private final ThreadLocal<Integer> threadId = ThreadLocal.withInitial(() -> (int) (Thread.currentThread().getId() % threads));
    private final ForkJoinPool customThreadPool = new ForkJoinPool(threads);

    public NeuralNetwork(Data data, List<LayerTemplate> tempLayers, double learningRate, long seed, int steps,
                         int batchSkip, double momentumAlpha, double rmsAlpha) {
        this.data = data;
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.random = new Random(seed);
        this.steps = steps;
        this.batch = batchSkip;
        this.momentumAlpha = momentumAlpha;
        this.rmsAlpha = rmsAlpha;
        initLayers(tempLayers);
    }

    private void initLayers(List<LayerTemplate> templateLayers) {
        for (int i = 0; i < templateLayers.size(); i++) {
            LayerTemplate layerTemplate = templateLayers.get(i);
            LayerTemplate layerTemplateNext = i == templateLayers.size() - 1 ? null : templateLayers.get(i + 1);
            layers.add(new Layer(
                    layerTemplate.getSize(),
                    layerTemplateNext == null ? 0 : layerTemplateNext.getSize(),
                    layerTemplate.getActivationFunction(), i == 0));
        }
        initializeWeights();
    }

    public void initializeWeights() {  // the Normal He-initialization
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

    public void train() throws Exception {  // Stochastic Gradient Descent
        int p = data.getTrainVectors().size();  // number of training examples
        int batchSize = Math.min(p, batch);
        List<Integer> batches = IntStream.rangeClosed(0, p - 1).boxed().collect(Collectors.toList());

        for (int t = 0; t < steps; t++) {
            Collections.shuffle(batches, random);  // random choice of the minibatch
            customThreadPool.submit(() -> batches.subList(0, batchSize).parallelStream().forEach(k -> {
                int tid = (int) (Thread.currentThread().getId() % threads);

                forward(data.getTrainVectors().get(k), tid);
                backpropagate(k, errorFunction, tid);
                computeGradient(tid);

            })).get();
            updateWeights();
        }
    }

    public <T extends Number> void forward(List<Double> input, int tid) {
        Layer inputLayer = layers.get(0);

        inputLayer.getOutputs()[tid][0] = 1; // bias
        for (int i = 0; i < inputLayer.getSize(); i++) {  // start the forward pass by evaluating the input neurons
            inputLayer.getOutputs()[tid][i + 1] = input.get(i);
        }

        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {
                double potential = 0;

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    potential += previousLayer.getWeights()[j][i] * previousLayer.getOutputs()[tid][i];
                }
                layer.getPotentials()[tid][j] = potential;
            }

            double max = Arrays.stream(layer.getPotentials()[tid]).max().orElse(0);

            // calculate sum of activation functions applied to potentials (for the output layer with softmax) - optimisation
            double sum = 0;
            for (int j = 0; j < layer.getSize(); j++) {
                sum += layer.getActivationFunction().apply(layer.getPotentials()[tid][j], max);
            }

            for (int j = 0; j < layer.getSize(); j++) {  // calculate outputs for each neuron in the layer
                layer.getOutputs()[tid][j + 1] = layer.getActivationFunction()
                        .computeOutput(sum, layer.getPotentials()[tid][j], max);
            }
        }
    }

    private void backpropagate(int k, ErrorFunction errorFunction, int tid) {
        Layer outputLayer = layers.get(layers.size() - 1);
        for (int j = 0; j < outputLayer.getSize(); j++) {  // start with the output layer
            double y = outputLayer.getOutputs()[tid][j + 1];
            double d = data.getTrainLabels().get(k).get(j);

            // compute the partial derivative of the errorFunction with respect to the outputs
            outputLayer.getChainRuleTermWithOutput()[tid][j] = errorFunction.calculatePartialDerivative(y, d);
        }

        // compute the partial derivative of the errorFunction with respect to the outputs in the hidden layers
        for (int l = layers.size() - 2; l >= 1; l--) {
            Layer nextLayer = layers.get(l + 1);
            Layer layer = layers.get(l);

            double max = Arrays.stream(nextLayer.getPotentials()[tid]).max().orElse(0);
            double sum = IntStream.range(0, nextLayer.getSize()).mapToDouble(j ->
                    nextLayer.getActivationFunction().apply(nextLayer.getPotentials()[tid][j], max)).sum();

            // sum of activation functions applied to potentials - optimisation

            for (int j = 0; j < layer.getSize(); j++) {
                layer.getChainRuleTermWithOutput()[tid][j] = 0;
            }
            for (int r = 0; r < nextLayer.getSize(); r++) {
                double term1 = nextLayer.getChainRuleTermWithOutput()[tid][r];
                double term2 = nextLayer.getActivationFunction().computeDerivative(sum, nextLayer.getPotentials()[tid][r], max);
                double t12 = term1 * term2;
                for (int j = 0; j < layer.getSize(); j++) {
                    double term3 = layer.getWeights()[r][j + 1];
                    layer.getChainRuleTermWithOutput()[tid][j] += t12 * term3;
                }
            }
        }
    }

    private void computeGradient(int tid) {
        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            double max = Arrays.stream(layer.getPotentials()[tid]).max().orElse(0);
            double sum = IntStream.range(0, layer.getSize()).mapToDouble(j ->
                    layer.getActivationFunction().apply(layer.getPotentials()[tid][j], max)).sum();


            for (int j = 0; j < layer.getSize(); j++) {
                double term1 = layer.getChainRuleTermWithOutput()[tid][j];
                double term2 = layer.getActivationFunction().computeDerivative(sum, layer.getPotentials()[tid][j], max);
                double t12 = term1 * term2;

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    double term3 = previousLayer.getOutputs()[tid][i];
                    double res = t12 * term3;

                    previousLayer.getWeightsStepAccumulator2()[tid][j][i] += res; // accumulate the partial derivatives
                }
            }
        }
    }

    private void updateWeights() {  // update the weights + optimize with momentum and RMSProp (Adam)
        double delta = 1e-8;  // smoothing term to avoid division by zero

        for (int l = 1; l < layers.size(); l++) {
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {
                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    // the big sum
                    // read-only so no mutex needed
                    double step = 0;
                    for (int tid = 0; tid < threads; tid++) {
                        step += previousLayer.getWeightsStepAccumulator2()[tid][j][i];
                    }

                    // r_ji^(t - 1)
                    double rmsProp = previousLayer.getRmsprop()[j][i];

                    // r_ji^(t)
                    double currentRmsProp = rmsAlpha * rmsProp + (1 - rmsAlpha) * step * step;

                    double actualStep = -(learningRate / Math.sqrt(currentRmsProp + delta)) * step;
                    double previousStep = previousLayer.getMomentum()[j][i];

                    double momentumBalancedStep = actualStep * (1 - momentumAlpha) + momentumAlpha * previousStep;

                    previousLayer.getWeights()[j][i] += momentumBalancedStep;
                    previousLayer.getRmsprop()[j][i] = currentRmsProp;
                    previousLayer.getMomentum()[j][i] = momentumBalancedStep;

                    for (int tid = 0; tid < threads; tid++) {
                        previousLayer.getWeightsStepAccumulator2()[tid][j][i] = 0;
                    }
                }
            }
        }
    }


    public void evaluate(String fileName) throws Exception {
        System.out.println("==============");
        File csvOutputFile = new File(fileName);
        int p = data.getTestVectors().size();
        int[] results = new int[p];

        customThreadPool.submit(() -> {
            IntStream.range(0, p).parallel().forEach(k -> {
                int tid = threadId.get();
                forward(data.getTestVectors().get(k), tid);
                Layer outputLayer = layers.get(layers.size() - 1);
                double max = -Double.MAX_VALUE;
                int result = 0;
                for (int j = 0; j < outputLayer.getSize(); j++) {
                    double predicted = outputLayer.getOutputs()[tid][j + 1];
                    if (predicted > max) {
                        max = predicted;
                        result = j;
                    }
                }
                results[k] = result;
            });
        }).get();

        try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
            for (int result : results) {
                pw.println(result);
            }
        }
    }
}
