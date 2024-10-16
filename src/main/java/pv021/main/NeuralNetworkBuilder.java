package pv021.main;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBuilder {
    private final List<LayerTemp> network;

    public NeuralNetworkBuilder(int inputSize) {
        this.network = new ArrayList<>();
        this.network.add(new LayerTemp(inputSize, new IdentityFunction()));
    }

    public NeuralNetworkBuilder addLayer(int n, ActivationFunction activationFunction) {
        network.add(new LayerTemp(n, activationFunction));
        return this;
    }

    public NeuralNetwork build(int outputSize, ActivationFunction activationFunction) {
        network.add(new LayerTemp(outputSize, activationFunction));
        return new NeuralNetwork();  // TODO
    }
}
