package pv021.main;

public class ReLuFunction extends ActivationFunction{

    @Override
    public float apply(float potential) {
        return potential > 0 ? potential : 0;
    }

    @Override
    public float applyDifferentiated(float potential) {
        return potential > 0 ? 1 : 0;
    }
}
