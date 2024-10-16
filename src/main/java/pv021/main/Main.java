package pv021.main;

import pv021.data.Data;
import pv021.function.activation.ReLuFunction;
import pv021.network.NeuralNetwork;
import pv021.network.builder.NeuralNetworkBuilder;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("Loading data...");
        Data data = new Data("data/fashion_mnist");

        System.out.println("Initialising Neural Network...");
        NeuralNetwork neuralNetwork = new NeuralNetworkBuilder(data).build(8, new ReLuFunction());

        System.out.println("Training...");
        // neuralNetwork.train(); TODO
        // evaluate and save test results
    }
}