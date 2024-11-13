package pv021.function.activation;

public class ReLuFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential) {
        return apply(potential);
    }

    @Override
    public double apply(double potential) {
        return potential > 0 ? potential : 0;
    }

    @Override
    public double computeDerivative(double sum, double potential) {
        return potential > 0 ? 1 : 0;
    }
}
