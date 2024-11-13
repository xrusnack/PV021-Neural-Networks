package pv021.function.activation;

public class TanHFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential) {
        return apply(potential);
    }

    @Override
    public double apply(double potential) {
        return Math.tanh(potential);
    }

    @Override
    public double computeDerivative(double sum, double potential) {
        return 1 - apply(potential) * apply(potential);
    }
}
