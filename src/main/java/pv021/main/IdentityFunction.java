package pv021.main;

public class IdentityFunction extends ActivationFunction{

    @Override
    public float apply(float potential) {
        return potential;
    }

    @Override
    public float applyDifferentiated(float potential) {
        return 0;
    }
}
