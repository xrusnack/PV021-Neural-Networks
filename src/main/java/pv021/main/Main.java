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
                0.00184,
                1440,
                2048,
                0.35,
                0.9,
                1,
                16)
                .addLayer(128, new ReLuFunction())
                .build(new SoftMaxFunction());

        System.out.println("Training...");
        neuralNetwork.train();

        // evaluate and save test results
        neuralNetwork.evaluate("predictions.csv");
    }
}