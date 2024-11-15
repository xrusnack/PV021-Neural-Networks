package pv021.network;

import pv021.function.activation.ActivationFunction;

/**
 * The class represents a single layer in a neural network, managing
 * its neurons, weights, activations, and connections to the next layer.
 */

public class Layer {

    private final double[][] outputs;
    private final double[][] potentials;

    private final double[][] weights;
    private final ActivationFunction activationFunction;
    private final int nextLayerSize;

    private final double[][] chainRuleTermWithOutput;


    private final Object weightsStepAccumulatorMutex = new Object();
    private final double[][][] weightsStepAccumulator;
    private final double[][] momentum;
    private final double[][] rmsprop;
    private final int size;

    public Layer(int size, int nextLayerSize, ActivationFunction activationFunction, boolean input) {
        this.nextLayerSize = nextLayerSize;
        this.activationFunction = activationFunction;
        this.size = size;

        this.outputs = new double[NeuralNetwork.threads][size + 1];
        this.potentials = input ? null : new double[NeuralNetwork.threads][size];
        chainRuleTermWithOutput = new double[NeuralNetwork.threads][size];

        // size + 1 to include bias
        this.weightsStepAccumulator = nextLayerSize > 0 ? new double[NeuralNetwork.threads][nextLayerSize][size + 1] : null;
        this.momentum = nextLayerSize > 0 ? new double[nextLayerSize][size + 1] : null;
        this.weights = nextLayerSize > 0 ? new double[nextLayerSize][size + 1] : null;
        this.rmsprop = nextLayerSize > 0 ? new double[nextLayerSize][size + 1] : null;
    }

    public int getSize() {
        return size;
    }

    public int getNextLayerSize() {
        return nextLayerSize;
    }

    public boolean isOutputLayer() {
        return weights == null;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public double[][] getPotentials() {
        return potentials;
    }

    public double[][] getOutputs() {
        return outputs;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[][][] getWeightsStepAccumulator2() {
        return weightsStepAccumulator;
    }

    public Object getWeightsStepAccumulatorMutex() {
        return weightsStepAccumulatorMutex;
    }

    public double[][] getChainRuleTermWithOutput() {
        return chainRuleTermWithOutput;
    }

    public static void main(String[] args) {
        double[][] tst = new double[4][4];
        tst[0][3] = 3;
        System.err.println(tst[0][3]);
    }

    public double[][] getMomentum() {
        return momentum;
    }

    public double[][] getRmsprop() {
        return rmsprop;
    }
}
