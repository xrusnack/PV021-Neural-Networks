package pv021.data;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class Data {
    private final List<List<Integer>> trainVectors;
    private final List<List<Integer>> testVectors;
    private final List<List<Integer>> trainLabels;
    private final List<List<Integer>> testLabels;
    private final int labelCount;

    public Data(String path, int labelCount) throws IOException {
        this.labelCount = labelCount;
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

    public List<List<Integer>> getTrainVectors() {
        return trainVectors;
    }

    public List<List<Integer>> getTestVectors() {
        return testVectors;
    }

    public List<List<Integer>> getTrainLabels() {
        return trainLabels;
    }

    public List<List<Integer>> getTestLabels() {
        return testLabels;
    }
}
