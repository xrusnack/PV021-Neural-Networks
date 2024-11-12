package pv021.function.activation;

public class IdentityFunction extends ActivationFunction {

    @Override
    public double computeOutput(double sum, double potential) {
        return apply(potential);
    }

    @Override
    public double apply(double potential) {
        return potential;
    }

    @Override
    public double applyDifferentiated(double potential) {
        return 0;
    }
}
