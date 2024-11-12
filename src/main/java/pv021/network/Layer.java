package pv021.network;

import pv021.function.activation.ActivationFunction;

public class Layer {
    private final double[] outputs;
    private final double[] potentials;
    private final double[] biases;
    private final double[][] weights;
    private final double[][] weightsUpdate;
    private final ActivationFunction activationFunction;
    private final int nextLayerSize;

    private final double[] chainRuleTermWithOutput;

    public Layer(int size, int nextLayerSize, ActivationFunction activationFunction) {
        this.outputs = new double[size];
        this.potentials = new double[size];
        this.biases = new double[size];
        this.chainRuleTermWithOutput = new double[size];
        this.activationFunction = activationFunction;
        this.nextLayerSize = nextLayerSize;

        // size + 1 to include bias
        this.weights = nextLayerSize > 0 ? new double[nextLayerSize][size] : null;
    }

    public int getSize(){
        return outputs.length;
    }

    public int getNextLayerSize() {
        return nextLayerSize;
    }

    public boolean isOutputLayer(){
        return weights == null;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public double[] getPotentials() {
        return potentials;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }

    public double[] getChainRuleTermWithOutput() {
        return chainRuleTermWithOutput;
    }
}
