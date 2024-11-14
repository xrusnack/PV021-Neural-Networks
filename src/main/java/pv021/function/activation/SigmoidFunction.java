package pv021.function.activation;

public class SigmoidFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential, double max) {
        return apply(potential, max);
    }

    @Override
    public double apply(double potential, double max) {
        return Math.pow(Math.E, potential) / (1 + Math.pow(Math.E, potential));
    }

    @Override
    public double computeDerivative(double sum, double potential, double max) {
        return (apply(potential, max) - 1) / (apply(potential, max));
    }
}
