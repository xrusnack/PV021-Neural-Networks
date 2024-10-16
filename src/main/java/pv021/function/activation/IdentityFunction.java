package pv021.function.activation;

public class IdentityFunction extends ActivationFunction {

    @Override
    public float apply(float potential) {
        return potential;
    }

    @Override
    public float applyDifferentiated(float potential) {
        return 0;
    }
}
