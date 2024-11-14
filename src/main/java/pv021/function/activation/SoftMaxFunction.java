package pv021.function.activation;

public class SoftMaxFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential, double max) {
        return apply(potential, max) / sum;
    }

    @Override
    public double apply(double potential, double max) {
        return Math.pow(Math.E, potential - max);
    }

    @Override
    public double computeDerivative(double sum, double potential, double max) {
        double x = computeOutput(sum, potential, max);
        return x * (1 - x);
    }
}
