package pv021.network;

import pv021.data.Data;
import pv021.function.error.CrossEntropy;
import pv021.function.error.ErrorFunction;
import pv021.network.builder.LayerTemp;

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
    private final double decay;
    private final double dropout;
    private final ErrorFunction errorFunction = new CrossEntropy();

    public static int threads = Runtime.getRuntime().availableProcessors();
    private final ThreadLocal<Integer> threadId = ThreadLocal.withInitial(() -> (int) (Thread.currentThread().getId() % threads));
    private final ForkJoinPool customThreadPool = new ForkJoinPool(threads);
    private final int evaluationStep;

    public NeuralNetwork(Data data, List<LayerTemp> tempLayers,
                         double learningRate, long seed, int steps, int batchSkip, double momentumAlpha, int evaluationStep,
                         double rmsAlpha, double decay, double dropout) {
        this.data = data;
        this.layers = new ArrayList<>();
        this.learningRate = learningRate;
        this.random = new Random(seed);
        this.steps = steps;
        this.batch = batchSkip;
        this.momentumAlpha = momentumAlpha;
        this.evaluationStep = evaluationStep;
        this.rmsAlpha = rmsAlpha;
        this.decay = decay;
        this.dropout = dropout;
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

    public void stochasticGradientDescent() throws Exception {
        int p = data.getTrainVectors().size();  // number of training examples
        int batchSize = Math.min(p, batch);
        List<Integer> batches = IntStream.rangeClosed(0, p - 1).boxed().collect(Collectors.toList()); // choose a minibatch

        for (int t = 0; t < steps; t++) {
            if (t % evaluationStep == 0 && t != 0) {
                printError(t);
            }
            Collections.shuffle(batches, random);
            customThreadPool.submit(() -> batches.subList(0, batchSize).parallelStream().forEach(k -> {
                int tid = (int) (Thread.currentThread().getId() % threads);
                forward(data.getTrainVectors().get(k), tid);
                backpropagate(k, errorFunction, tid);
                computeGradient(tid);
            })).get();
            updateWeights();
            //System.err.println(t);
            /*System.err.println(t + " | max = %f min = %f".formatted(
                    Arrays.stream(layers.get(layers.size() - 1).getPotentials()).max().orElse(0),
                    Arrays.stream(layers.get(layers.size() - 1).getPotentials()).min().orElse(0)));*/
        }

        if (steps >= evaluationStep) {
            printError(steps);
        }
    }

    public <T extends Number> void forward(List<Double> input, int tid) {
        Layer inputLayer = layers.get(0);

        inputLayer.getOutputs()[tid][0] = 1; // bias
        for (int i = 0; i < inputLayer.getSize(); i++) {  // start the forward pass by evaluating the input neurons
            inputLayer.getOutputs()[tid][i + 1] = input.get(i);
        }

        for (int l = 1; l < layers.size(); l++) {  // forward pass in the hidden layers
            Layer previousLayer = layers.get(l - 1);
            Layer layer = layers.get(l);

            for (int j = 0; j < layer.getSize(); j++) {
                double potential = 0;

                for (int i = 0; i < previousLayer.getSize() + 1; i++) {
                    double w = previousLayer.getWeights()[j][i];
                    double o = previousLayer.getOutputs()[tid][i];
                    potential += w * o;
                }

                layer.getPotentials()[tid][j] = potential;

            }
            double max = Arrays.stream(layer.getPotentials()[tid]).max().orElse(0);

            // calculate sum of activation functions applied to potentials (for the output layer with softmax) - optimisation
            double sum = 0;
            for (int j = 0; j < layer.getSize(); j++) {
                sum += layer.getActivationFunction().apply(layer.getPotentials()[tid][j], max);
            }

            // calculate outputs for each neuron in the layer
            for (int j = 0; j < layer.getSize(); j++) {
                layer.getOutputs()[tid][j + 1] = layer.getActivationFunction()
                        .computeOutput(sum, layer.getPotentials()[tid][j], max);/* * layer.getDropout()[tid][j]; */
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

                    previousLayer.getWeightsStepAccumulator2()[tid][j][i] += res;/* * layer.getDropout()[tid][j]; */  // accumulate the partial derivatives
                }
            }
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

                    previousLayer.getWeights()[j][i] *= (1 - decay);
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


    private void printError(int step) throws Exception {
        Layer outputLayer = layers.get(layers.size() - 1);

        AtomicReference<Double> errorTestRef = new AtomicReference<>(0.0);
        final AtomicInteger total = new AtomicInteger(0);
        final AtomicInteger correct = new AtomicInteger(0);

        final int p = data.getTestVectors().size();
        customThreadPool.submit(() -> {
            IntStream.range(0, p).parallel().forEach(k -> {
                int tid = threadId.get();
                forward(data.getTestVectors().get(k), tid);
                double max = -Double.MAX_VALUE;
                int result = 0;
                double errorAccumulator = 0;
                for (int j = 0; j < outputLayer.getSize(); j++) {
                    double predicted = outputLayer.getOutputs()[tid][j + 1];
                    int truth = data.getTestLabels().get(k).get(j);
                    if (predicted > max) {
                        max = predicted;
                        result = j;
                    }

                    errorAccumulator -= (truth * Math.log(predicted) + (1 - truth) * Math.log(1 - predicted)) / p;
                }
                double finalErrorAccumulator = errorAccumulator;
                errorTestRef.updateAndGet(v -> v + finalErrorAccumulator);
                total.incrementAndGet();
                if (data.getTestLabels().get(k).get(result) == 1) {
                    correct.incrementAndGet();
                }
            });
        }).get();

        double accuracyTest = (correct.get() * 100.0) / total.get();

        AtomicReference<Double> errorTrainRef = new AtomicReference<>(0.0);
        total.set(0);
        correct.set(0);

        final int p2 = data.getTrainVectors().size();
        customThreadPool.submit(() -> IntStream.range(0, p2).parallel().forEach(k -> {
            int tid = threadId.get();
            forward(data.getTrainVectors().get(k), tid);
            double max = -Double.MAX_VALUE;
            int result = 0;
            double errorAccumulator = 0;
            for (int j = 0; j < outputLayer.getSize(); j++) {
                double predicted = outputLayer.getOutputs()[tid][j + 1];
                int truth = data.getTrainLabels().get(k).get(j);
                if (predicted > max) {
                    max = predicted;
                    result = j;
                }

                errorAccumulator -= (truth * Math.log(predicted) + (1 - truth) * Math.log(1 - predicted)) / p2;
            }
            double finalErrorAccumulator = errorAccumulator;
            errorTrainRef.updateAndGet(v -> v + finalErrorAccumulator);
            total.incrementAndGet();
            if (data.getTrainLabels().get(k).get(result) == 1) {
                correct.incrementAndGet();
            }
        })).get();

        double errorTrain = errorTrainRef.get();
        double errorTest = errorTestRef.get();

        double accuracyTrain = (correct.get() * 100.0) / total.get();
        System.out.printf("Step %d: Test: [loss: %.6f accuracy: %.2f%%] Train: [loss: %.6f accuracy: %.2f%%] Overfit: %.6f / %.2f%%%n", step, errorTest, accuracyTest, errorTrain, accuracyTrain, -errorTrain + errorTest, accuracyTrain - accuracyTest);
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
