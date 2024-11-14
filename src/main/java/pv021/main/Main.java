package pv021.main;

import pv021.data.Data;
import pv021.function.activation.ReLuFunction;
import pv021.function.activation.SigmoidFunction;
import pv021.function.activation.SoftMaxFunction;
import pv021.function.activation.TanHFunction;
import pv021.network.NeuralNetwork;
import pv021.network.builder.NeuralNetworkBuilder;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading data...");
        Data data = new Data("data/fashion_mnist", 10);
        //Data data = new Data("data/xor", 2);
        //Data data = new Data("data/export", 2);

        System.out.println("Initialising Neural Network...");
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder(data,
                0.0000075,
                1000,
                128,
                0.96,
                0,
                false)
                .addLayer(3000, new ReLuFunction())
                .build(new SoftMaxFunction());

        System.out.println("Initialising Neural Weights...");
        neuralNetwork.initializeWeights();

        System.out.println("Training...");
        neuralNetwork.trainBatch();
        // evaluate and save test results

        neuralNetwork.evaluate("predictions.csv");

        // XOR only
        //neuralNetwork.drawDistribution();
    }
}