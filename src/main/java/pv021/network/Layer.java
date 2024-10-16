package pv021.network;

import pv021.function.activation.ActivationFunction;

public class Layer {
    private final double[] outputs;
    private final double[] potentials;
    private final double[][] weights;
    private final ActivationFunction activationFunction;

    public Layer(int size, int nextLayerSize, ActivationFunction activationFunction) {
        this.outputs = new double[size];
        this.potentials = new double[size];
        this.activationFunction = activationFunction;
        this.weights = nextLayerSize > 0 ? new double[size][nextLayerSize] : null;
    }

    public int getSize(){
        return outputs.length;
    }

    public boolean isOutputLayer(){
        return weights == null;
    }
}
