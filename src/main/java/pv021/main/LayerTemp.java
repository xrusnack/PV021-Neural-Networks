package pv021.main;

public class LayerTemp {
    private final int size;
    private final ActivationFunction activationFunction;

    public LayerTemp(int size, ActivationFunction activationFunction) {
        this.size = size;
        this.activationFunction = activationFunction;
    }

    public int getSize() {
        return size;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
