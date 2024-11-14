package pv021.data;

import java.io.IOException;
import java.io.DataInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * The class represents a dataset containing training and testing vectors and labels.
 */

public class Data {
    private final List<List<Double>> trainVectors;
    private final List<List<Double>> testVectors;
    private final List<List<Integer>> trainLabels;
    private final List<List<Integer>> testLabels;
    private final int labelCount;

    public Data(String path, int labelCount) throws IOException {
        this.labelCount = labelCount;
        this.trainVectors = normalizeVectors(loadVectors(path, true));
        this.testVectors = normalizeVectors(loadVectors(path, false));
        this.testLabels = loadLabels(path, false);
        this.trainLabels = loadLabels(path, true);
        checkData();
    }

    private void checkData() {
        if(trainVectors.isEmpty()){
            throw new IllegalStateException("Train Vectors are empty!");
        }

        if(testVectors.isEmpty()){
            throw new IllegalStateException("Test Vectors are empty!");
        }

        if(trainVectors.size() != trainLabels.size()){
            throw new IllegalStateException("Train Vectors size is not equal to Train Labels size!");
        }

        if(testLabels.size() != testVectors.size()){
            throw new IllegalStateException("Test Vectors size is not equal to Test Labels size!");
        }

        if(!trainVectors.stream().allMatch(integers -> integers.size() == trainVectors.get(0).size())){
            throw new IllegalStateException("Train Vectors have inconsistent size!");
        }

        if(!testVectors.stream().allMatch(integers -> integers.size() == testVectors.get(0).size())){
            throw new IllegalStateException("Test Vectors have inconsistent size!");
        }

        if(testVectors.get(0).size() != trainVectors.get(0).size()){
            throw new IllegalStateException("Train Vectors have different size than Test vectors!");
        }
    }

    public int getLabelCount() {
        return labelCount;
    }

    private List<List<Integer>> loadVectors(String path, boolean train) throws IOException {
        String vectorsPath = path + (train ? "_train_vectors.csv" : "_test_vectors.csv");

        try(DataInputStream in = new DataInputStream(new FileInputStream(vectorsPath))){
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(in));
            List<List<Integer>> result = new ArrayList<>();
            String line;

            while ((line = bufferedReader.readLine()) != null) {
                List<Integer> vector = new ArrayList<>();
                String[] numbers = line.split(",");

                for (String element : numbers) {
                    int elementInt = Integer.parseInt(element);
                    vector.add(elementInt);
                }
                result.add(vector);
            }
            return result;
        }
    }

    private List<List<Double>> normalizeVectors(List<List<Integer>> vectors) {
        double mean = calculateMean(vectors);
        double std  = calculateStandardDeviation(vectors, mean);

        List<List<Double>> normalizedVectors = new ArrayList<>();

        for (List<Integer> vector : vectors) {
            List<Double> normalizedVector = new ArrayList<>();
            for (int value : vector) {
                normalizedVector.add((value - mean) / std);
            }
            normalizedVectors.add(normalizedVector);
        }
        return normalizedVectors;
    }

    private double calculateMean(List<List<Integer>> vectors) {
        return vectors.stream()
                .flatMapToInt(list -> list.stream().mapToInt(Integer::intValue))
                .average()
                .orElse(0.0);
    }

    private double calculateStandardDeviation(List<List<Integer>> vectors, double mean) {
        long count = vectors.stream().mapToLong(List::size).sum();

        double sumOfSquares = vectors.stream()
                .flatMapToInt(list -> list.stream().mapToInt(Integer::intValue))
                .mapToDouble(value -> Math.pow(value - mean, 2))
                .sum();

        return Math.sqrt(sumOfSquares / (count - 1));
    }

    private List<List<Integer>> loadLabels(String path, boolean train) throws IOException {
        String vectorsPath = path + (train ? "_train_labels.csv" : "_test_labels.csv");
        try(DataInputStream in = new DataInputStream(new FileInputStream(vectorsPath))){
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(in));

            List<List<Integer>> result = new ArrayList<>();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                int value = Integer.parseInt(line);
                result.add(oneHotEncode(value, labelCount));
            }
            return result;
        }
    }

    private List<Integer> oneHotEncode(int value, int labelCount) {
        List<Integer> result = new ArrayList<>();
        for(int i = 0; i < labelCount; i++){
            result.add(i == value ? 1 : 0);
        }

        return result;
    }

    public List<List<Double>> getTrainVectors() {
        return trainVectors;
    }

    public List<List<Double>> getTestVectors() {
        return testVectors;
    }

    public List<List<Integer>> getTrainLabels() {
        return trainLabels;
    }

    public List<List<Integer>> getTestLabels() {
        return testLabels;
    }
}
