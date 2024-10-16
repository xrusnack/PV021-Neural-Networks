package pv021.main;

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