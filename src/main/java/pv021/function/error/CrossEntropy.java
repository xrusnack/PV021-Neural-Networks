package pv021.function.error;

public class CrossEntropy extends ErrorFunction {
    @Override
    public double calculatePartialDerivative(double output, double label) {
        return -label / output + (1 - label) / (1 - output);};

}
