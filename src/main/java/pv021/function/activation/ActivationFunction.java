package pv021.function.activation;

public abstract class ActivationFunction {
    public abstract float apply(float potential);
    public abstract float applyDifferentiated(float potential);
}
