package pv021.network.builder;

import pv021.function.activation.ActivationFunction;
import pv021.function.activation.IdentityFunction;
import pv021.data.Data;
import pv021.network.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkBuilder {
    private final List<LayerTemp> layers;
    private final Data data;
    private final double learningRate;
    private final int steps;
    private final long seed;
    private final int batchSkip;
    private final double momentumAlpha;

    public NeuralNetworkBuilder(Data data, double learningRate, int steps, int batchSkip, double momentumAlpha, long seed) {
        this.layers = new ArrayList<>();
        this.data = data;
        this.steps = steps;
        this.learningRate = learningRate;
        this.batchSkip = batchSkip;
        this.momentumAlpha = momentumAlpha;
        this.layers.add(new LayerTemp(data.getTestVectors().get(0).size(), new IdentityFunction()));
        this.seed = seed;
    }

    public NeuralNetworkBuilder addLayer(int n, ActivationFunction activationFunction) {
        layers.add(new LayerTemp(n, activationFunction));
        return this;
    }

    public NeuralNetwork build(ActivationFunction activationFunction) {
        layers.add(new LayerTemp(data.getLabelCount(), activationFunction));
        return new NeuralNetwork(data, layers, learningRate, seed, steps, batchSkip, momentumAlpha);
    }
}
