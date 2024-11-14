package pv021.function.activation;

public class ReLuFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential, double max) {
        return apply(potential, max);
    }

    @Override
    public double apply(double potential, double max) {
        return potential > 0 ? potential : 0;
    }

    @Override
    public double computeDerivative(double sum, double potential, double max) {
        return potential > 0 ? 1 : 0;
    }
}
