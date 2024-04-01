public class MultiPerceptronV2 {
    // Stores the weights vectors for each class.
    private double[] perceptrons;
    private int numClasses; // Number m of classes in the MultiPerceptron

    // Creates a multi-perceptron object with m classes and n inputs.
    // It creates an array of m perceptrons, each with n inputs.
    public MultiPerceptronV2(int m, int n) {
        perceptrons = new double[n * m];
        numClasses = m;
    }

    // Returns the number of classes m.
    public int numberOfClasses() {
        return numClasses;
    }

    // Returns the number of inputs n (length of the feature vector).
    public int numberOfInputs() {
        return perceptrons.length / numClasses;
    }

    // Returns the predicted class label (between 0 and m-1) for the given input.
    public int predictMulti(double[] x) {
        int classPrediction = 0;
        int index = 0;
        int weightedSum = 0;

        // First loop repeats as many times as the number of classes.
        for (int i = 0; i < numClasses; i++) {
            // Resets sum for every class.
            int tempSum = 0;

            // Second loop iterates through every element in array x.
            for (int j = 0; j < x.length; j++) {
                // Stores a temporary sum and compares to the largest weighted
                // sum that has occurred, updating the prediction.
                tempSum += x[j] * perceptrons[j + index];
            }
            if (i == 0) weightedSum = tempSum;
            if (tempSum > weightedSum) {
                classPrediction = i;
                weightedSum = tempSum;
            }
            index += numberOfInputs(); // Increment the index for each class.
        }
        return classPrediction;
    }

    // Trains this multi-perceptron on the labeled (between 0 and m-1) input.
    public void trainMulti(double[] x, int classLabel) {
        // int[] prediction = new int[numClasses];

        // First, calculate weighted sums:
        int index = 0;

        // first for loop iterates through m times
        for (int i = 0; i < numClasses; i++) {
            int sums = 0;
            double prediction = -1;

            // store the weighted sum for each class
            for (int j = 0; j < x.length; j++) {
                sums += x[j] * perceptrons[j + index];
            }
            // we have now created the weights vectors

            // Second: Get prediction based on positivity:
            if (sums > 0) prediction = 1;

            for (int k = 0; k < numberOfInputs(); k++) {
                // if false negative prediction, change weights values of the class
                if (prediction == -1 && i == classLabel)
                    perceptrons[k + index] = perceptrons[k + index] + x[k];

                // if false positive prediction, change weights values of the class
                if (prediction == 1 && i != classLabel) {
                    perceptrons[k + index] = perceptrons[k + index] - x[k];
                }
            }
            // increments the indices of the perceptrons array based on class
            index += numberOfInputs();
        }
    }

    // Returns a String representation of this MultiPerceptron, with
    // the string representations of the perceptrons separated by commas
    // and enclosed in parentheses.
    // Example with m = 2 and n = 3: ((2.0, 0.0, -2.0), (3.0, 4.0, 5.0))
    public String toString() {
        String representation = "(";
        int index = 0;

        for (int i = 0; i < numClasses; i++) {
            representation += "(";
            for (int j = 0; j < numberOfInputs(); j++) {
                representation += perceptrons[j + index] + ", ";
            }
            representation += ")";
            index += numberOfInputs();
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

        MultiPerceptronV2 perceptron = new MultiPerceptronV2(m, n);
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

