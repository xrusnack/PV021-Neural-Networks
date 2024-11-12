package pv021.main;

import pv021.data.Data;
import pv021.function.activation.ReLuFunction;
import pv021.network.NeuralNetwork;
import pv021.network.builder.NeuralNetworkBuilder;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading data...");
        Data data = new Data("data/xor", 2);

        System.out.println("Initialising Neural Network...");
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder(data, 0.01)
                .addLayer(5, new ReLuFunction())
                .build(new ReLuFunction());

        System.out.println("Initialising Neural Weights...");
        neuralNetwork.initializeWeights();

        System.out.println("Training...");
        neuralNetwork.trainOnline();
        // evaluate and save test results
    }
}