package pv021.function.activation;

public abstract class ActivationFunction {
    public abstract double computeOutput(double sum, double potential, double max);
    public abstract double apply(double potential, double max);
    public abstract double computeDerivative(double sum, double potential, double max);
}
