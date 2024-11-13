package pv021.network;

import pv021.function.activation.ActivationFunction;

public class Layer {
    private final double[] outputs;
    private final double[] potentials;
    private final double[][] weights;
    private final ActivationFunction activationFunction;
    private final int nextLayerSize;

    private final double[] chainRuleTermWithOutput;
    private final double[][] weightsStepAccumulator;
    private final double[][] momentum;

    public Layer(int size, int nextLayerSize, ActivationFunction activationFunction) {
        this.outputs = new double[size + 1];
        outputs[0] = 1; // bias
        this.potentials = new double[size];
        this.chainRuleTermWithOutput = new double[size];
        this.activationFunction = activationFunction;
        this.nextLayerSize = nextLayerSize;

        // TODO! size + 1 to include bias
        this.weightsStepAccumulator = nextLayerSize > 0 ? new double[nextLayerSize][size + 1] : null;
        this.momentum = nextLayerSize > 0 ? new double[nextLayerSize][size + 1] : null;
        this.weights = nextLayerSize > 0 ? new double[nextLayerSize][size + 1] : null;
    }

    public int getSize(){
        return potentials.length;
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

    public double[][] getWeightsStepAccumulator() {
        return weightsStepAccumulator;
    }

    public double[] getChainRuleTermWithOutput() {
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
}
