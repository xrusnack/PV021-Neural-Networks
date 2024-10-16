package pv021.data;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class Data {
    private final List<Vector<Integer>> trainVectors;
    private final List<Vector<Integer>> testVectors;
    private final List<Integer> trainLabels;
    private final List<Integer> testLabels;

    public Data(String path) throws IOException {
        this.trainVectors = loadVectors(path, true);
        this.testVectors = loadVectors(path, false);
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

    private static List<Vector<Integer>> loadVectors(String path, boolean train) throws IOException {
        String vectorsPath = path + (train ? "_train_vectors.csv" : "_test_vectors.csv");
        try(DataInputStream in = new DataInputStream(new FileInputStream(vectorsPath))){
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(in));
            List<Vector<Integer>> result = new ArrayList<>();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                Vector<Integer> vector = new Vector<>();
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

    private static List<Integer> loadLabels(String path, boolean train) throws IOException {
        String vectorsPath = path + (train ? "_train_labels.csv" : "_test_labels.csv");
        try(DataInputStream in = new DataInputStream(new FileInputStream(vectorsPath))){
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(in));

            List<Integer> result = new ArrayList<>();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                result.add(Integer.parseInt(line));
            }
            return result;
        }
    }

    public List<Vector<Integer>> getTrainVectors() {
        return trainVectors;
    }

    public List<Vector<Integer>> getTestVectors() {
        return testVectors;
    }

    public List<Integer> getTrainLabels() {
        return trainLabels;
    }

    public List<Integer> getTestLabels() {
        return testLabels;
    }
}