package pv021.function.activation;

public class TanHFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential, double max) {
        return apply(potential, max);
    }

    @Override
    public double apply(double potential, double max) {
        return Math.tanh(potential);
    }

    @Override
    public double computeDerivative(double sum, double potential, double max) {
        return 1 - apply(potential, max) * apply(potential, max);
    }
}
