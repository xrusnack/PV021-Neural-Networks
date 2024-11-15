package pv021.main;

import pv021.data.Data;
import pv021.function.activation.ReLuFunction;
import pv021.function.activation.SoftMaxFunction;
import pv021.network.NeuralNetwork;
import pv021.network.builder.NeuralNetworkBuilder;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading data...");
        Data data = new Data("data/fashion_mnist", 10);

        System.out.println("Initialising Neural Network...");

        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder(data,
                0.0002,
                499,
                128,
                0.8,
                0.8, // 0.5: 88.84 0.8: 89.02
                42,
                0, // 0: 89.02 0.0001: 88.66 0.001: 85.69
                true)
                //.addLayer(4096, new ReLuFunction())
                .addLayer(512, new ReLuFunction())
                .build(new SoftMaxFunction());

        System.out.println("Training...");
        neuralNetwork.stochasticGradientDescent();
        // evaluate and save test results

        neuralNetwork.evaluate("predictions.csv");

        // XOR only
        //neuralNetwork.drawDistribution();
    }
}