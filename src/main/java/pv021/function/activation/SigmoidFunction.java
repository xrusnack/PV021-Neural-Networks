package pv021.function.activation;

public class SigmoidFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential) {
        return apply(potential);
    }

    @Override
    public double apply(double potential) {
        return Math.pow(Math.E, potential) / (1 + Math.pow(Math.E, potential));
    }

    @Override
    public double computeDerivative(double sum, double potential) {
        return (apply(potential) - 1) / (apply(potential));
    }
}
