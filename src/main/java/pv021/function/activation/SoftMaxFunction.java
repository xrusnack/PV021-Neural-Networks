package pv021.function.activation;

public class SoftMaxFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential) {
        return apply(potential) / sum;
    }

    @Override
    public double apply(double potential) {
        return Math.pow(Math.E, potential);
    }

    @Override
    public double computeDerivative(double sum, double potential) {
        double x = computeOutput(sum, potential);
        return x * (1 - x);
    }
}
