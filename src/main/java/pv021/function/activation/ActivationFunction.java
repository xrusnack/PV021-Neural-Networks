package pv021.function.activation;

public abstract class ActivationFunction {
    public abstract double computeOutput(double sum, double potential);
    public abstract double apply(double potential);
    public abstract double computeDerivative(double sum, double potential);
}
