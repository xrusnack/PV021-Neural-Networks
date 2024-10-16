package pv021.main;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        try {
            Data data = new Data("data/fashion_mnist");
        }catch(IOException e){
            e.printStackTrace();
        }
    }
}