package pv021.network.builder;

import pv021.function.activation.ActivationFunction;

public class LayerTemplate {
    private final int size;
    private final ActivationFunction activationFunction;

    public LayerTemplate(int size, ActivationFunction activationFunction) {
        this.size = size;
        this.activationFunction = activationFunction;
    }

    public int getSize() {
        return size;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
