public class Perceptron {
    private double[] weights; // stores the weight vector

    // Creates a perceptron with n inputs. It should create an array
    // of n weights and initialize them all to 0.
    public Perceptron(int n) {
        weights = new double[n];
    }

    // Returns the number of inputs n.
    public int numberOfInputs() {
        return weights.length;
    }

    // Returns the weighted sum of the weight vector and x.
    public double weightedSum(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * weights[i];
        }
        return sum;
    }

    // Predicts the binary label (+1 or -1) of input x. It returns +1
    // if the weighted sum is positive and -1 if it is negative (or zero).
    public int predict(double[] x) {
        if (weightedSum(x) > 0) return 1;
        return -1;
    }

    // Trains this perceptron on the binary labeled (+1 or -1) input x.
    // The weights vector is updated accordingly.
    public void train(double[] x, int binaryLabel) {
        int prediction = predict(x);
        for (int i = 0; i < x.length; i++) {

            if (binaryLabel == 1 && prediction == -1) // false negative
                weights[i] = weights[i] + x[i];

            if (binaryLabel == -1 && prediction == 1) // false positive
                weights[i] = weights[i] - x[i];
        }
    }

    // Returns a String representation of the weight vector, with the
    // individual weights separated by commas and enclosed in parentheses.
    // Example: (2.0, 1.0, -1.0, 5.0, 3.0)
    public String toString() {
        String representation = "(";
        for (int i = 0; i < weights.length; i++) {
            if (i == (weights.length - 1)) representation += weights[i] + ")";
            else representation += weights[i] + ", ";
        }
        return representation;
    }

    // Tests this class by directly calling all instance methods.
    public static void main(String[] args) {
        int n = 3;

        double[] training1 = { 3.0, 4.0, 5.0 };  // yes
        double[] training2 = { 2.0, 0.0, -2.0 };  // no
        double[] training3 = { -2.0, 0.0, 2.0 };  // yes
        double[] training4 = { 5.0, 4.0, 3.0 };  // no


        // Training data
        Perceptron perceptron = new Perceptron(n);
        StdOut.println("Number Of Inputs: " + perceptron.numberOfInputs());
        StdOut.println(perceptron);
        perceptron.train(training1, +1);
        StdOut.println(perceptron);
        perceptron.train(training2, -1);
        StdOut.println(perceptron);
        perceptron.train(training3, +1);
        StdOut.println(perceptron);

        // Calls each method class and prints to console.
        System.out.println("For set (5, 4, 3):");
        System.out.println("Binary label: -1");
        System.out.println("Weighted Sum: " + perceptron.weightedSum(training4));
        System.out.println("Prediction: " + perceptron.predict(training4));

        perceptron.train(training4, -1);
        StdOut.println("Adjusted weights: " + perceptron);
    }
}
