package pv021.main;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBuilder {
    private final List<LayerTemp> layers;
    private final Data data;

    public NeuralNetworkBuilder(Data data) {
        this.layers = new ArrayList<>();
        this.data = data;
        this.layers.add(new LayerTemp(data.getTestVectors().get(0).size(), new IdentityFunction()));
    }

    public NeuralNetworkBuilder addLayer(int n, ActivationFunction activationFunction) {
        layers.add(new LayerTemp(n, activationFunction));
        return this;
    }

    public NeuralNetwork build(int outputSize, ActivationFunction activationFunction) {
        layers.add(new LayerTemp(outputSize, activationFunction));
        return new NeuralNetwork(data, layers);  // TODO
    }
}
