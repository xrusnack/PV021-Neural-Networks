package pv021.function.activation;

public abstract class ActivationFunction {
    public abstract double apply(double potential);
    public abstract double applyDifferentiated(double potential);
}
