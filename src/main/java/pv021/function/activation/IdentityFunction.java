package pv021.function.activation;

public class IdentityFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential, double max) {
        return apply(potential, max);
    }

    @Override
    public double apply(double potential, double max) {
        return potential;
    }

    @Override
    public double computeDerivative(double sum, double potential, double max) {
        return 0;
    }
}
