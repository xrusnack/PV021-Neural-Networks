package pv021.function.error;

public class SquaredErrorFunction extends ErrorFunction{
    public double calculateError(double[] expected, double[] actual) {
        double result = 0.0;

        for(int i = 0; i < expected.length; i++){
            result += (expected[i] - actual[i]) * (expected[i] - actual[i]);
        }

        return result / 2.0;
    }

    @Override
    public double calculatePartialDerivative(double output, double label) { return output - label;};
}
