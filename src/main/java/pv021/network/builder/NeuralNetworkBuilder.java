package pv021.network.builder;

import pv021.function.activation.ActivationFunction;
import pv021.function.activation.IdentityFunction;
import pv021.data.Data;
import pv021.network.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

/**
 * A builder class for constructing a NeuralNetwork.
 */
public class NeuralNetworkBuilder {
    private final List<LayerTemplate> layers;
    private final Data data;
    private final double learningRate;
    private final int steps;
    private final long seed;
    private final int batchSkip;
    private final double momentumAlpha;
    private final double rmsAlpha;
    private final int threads;

    public NeuralNetworkBuilder(Data data, double learningRate, int steps, int batchSize, double momentumAlpha,
                                double rmsAlpha, long seed, int threads) {
        this.layers = new ArrayList<>();
        this.data = data;
        this.steps = steps;
        this.learningRate = learningRate;
        this.batchSkip = batchSize;
        this.momentumAlpha = momentumAlpha;
        this.layers.add(new LayerTemplate(data.getTestVectors().get(0).size(), new IdentityFunction()));
        this.seed = seed;
        this.rmsAlpha=rmsAlpha;
        this.threads = threads;
    }

    public NeuralNetworkBuilder addLayer(int n, ActivationFunction activationFunction) {
        layers.add(new LayerTemplate(n, activationFunction));
        return this;
    }

    public NeuralNetwork build(ActivationFunction activationFunction) {
        layers.add(new LayerTemplate(data.getLabelCount(), activationFunction));
        return new NeuralNetwork(data, layers, learningRate, seed, steps, batchSkip,
                momentumAlpha, rmsAlpha, threads);
    }

}
