public class MultiPerceptron {
    // Stores the weights vectors for each class.
    private Perceptron[] perceptrons;
    private int numClasses; // Number m of classes in the MultiPerceptron
    private int n; // Length of the input feature vector.

    // Creates a multi-perceptron object with m classes and n inputs.
    // It creates an array of m perceptrons, each with n inputs.
    public MultiPerceptron(int m, int n) {
        numClasses = m;
        this.n = n;

        perceptrons = new Perceptron[m];
        for (int i = 0; i < m; i++) {
            perceptrons[i] = new Perceptron(n);
        }
    }

    // Returns the number of classes m.
    public int numberOfClasses() {
        return numClasses;
    }

    // Returns the number of inputs n (length of the feature vector).
    public int numberOfInputs() {
        return n;
    }

    // Returns the predicted class label (between 0 and m-1) for the given input.
    public int predictMulti(double[] x) {
        int classPrediction = 0;
        double weightedSum = perceptrons[0].weightedSum(x);

        // Iterates through every element in the perceptrons array.
        for (int i = 0; i < numClasses; i++) {
            // Stores the weighted sum of each perceptron and compares.
            double tempSum = perceptrons[i].weightedSum(x);

            // Stores the class prediction and largest weighted sum.
            if (tempSum > weightedSum) {
                classPrediction = i;
                weightedSum = tempSum;
            }
        }
        return classPrediction;
    }

    // Trains this multi-perceptron on the labeled (between 0 and m-1) input.
    public void trainMulti(double[] x, int classLabel) {
        // Iterates through the MultiPerceptron array.
        for (int i = 0; i < numClasses; i++) {
            // if false negative prediction, change weights values of the class
            if (i == classLabel)
                perceptrons[i].train(x, 1);

            // if false positive prediction, change weights values of the class
            if (i != classLabel)
                perceptrons[i].train(x, -1);
        }
    }

    // Returns a String representation of this MultiPerceptron, with
    // the string representations of the perceptrons separated by commas
    // and enclosed in parentheses.
    // Example with m = 2 and n = 3: ((2.0, 0.0, -2.0), (3.0, 4.0, 5.0))
    public String toString() {
        String representation = "(";
        for (int i = 0; i < numClasses; i++) {
            if (i == (numClasses - 1))
                representation += perceptrons[i].toString();
            else
                representation += perceptrons[i].toString() + ", ";
        }
        return representation + ")";
    }

    // Tests this class by directly calling all instance methods.
    public static void main(String[] args) {
        int m = 2;
        int n = 3;

        // Training the MultiPerceptron.
        double[] training1 = { 3.0, 4.0, 5.0 };  // class 1
        double[] training2 = { 2.0, 0.0, -2.0 };  // class 0
        double[] training3 = { -2.0, 0.0, 2.0 };  // class 1
        double[] training4 = { 5.0, 4.0, 3.0 };  // class 0

        MultiPerceptron perceptron = new MultiPerceptron(m, n);
        StdOut.println(perceptron);
        perceptron.trainMulti(training1, 1);
        StdOut.println(perceptron);
        perceptron.trainMulti(training2, 0);
        StdOut.println(perceptron);
        perceptron.trainMulti(training3, 1);
        StdOut.println(perceptron);
        perceptron.trainMulti(training4, 0);
        StdOut.println(perceptron);

        StdOut.println("Number of Classes: " + perceptron.numberOfClasses());
        StdOut.println("Number of Inputs: " + perceptron.numberOfInputs());

        // Testing the MultiPerceptron.
        double[] testing1 = { -1.0, -2.0, 3.0 };
        double[] testing2 = { 2.0, -5.0, 1.0 };

        StdOut.println(perceptron.predictMulti(testing1));
        StdOut.println(perceptron);
        StdOut.println(perceptron.predictMulti(testing2));
        StdOut.println(perceptron);
    }
}
