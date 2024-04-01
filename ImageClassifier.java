import java.awt.Color;

public class ImageClassifier {
    private int n; // Stores the number of inputs.
    private int m; // Stores the number of classes.
    private String[] classNames; // Stores the names of all classes.
    // Creates a MultiPerceptron to classify images.
    private MultiPerceptron imageClassifier;

    // Uses the provided configuration file to create an
    // ImageClassifier object.
    public ImageClassifier(String configFile) {
        // Use the In library to read the configuration file.
        In in = new In(configFile);
        n = in.readInt() * in.readInt();
        m = in.readInt();
        imageClassifier = new MultiPerceptron(m, n);

        // Create an array to store the class names.
        classNames = new String[m];
        for (int i = 0; i < m; i++) {
            classNames[i] = in.readString();
        }
    }

    // Creates a feature vector (1D array) from the given picture.
    public double[] extractFeatures(Picture picture) {
        if ((picture.height() * picture.width() != n))
            throw new IllegalArgumentException("Image dimensions are not equal "
                                                       + "to dimensions provided"
                                                       + " in configuration file"
                                                       + ".");

        double[] fV = new double[picture.width() * picture.height()];
        int index = 0;
        for (int row = 0; row < picture.height(); row++) {
            for (int col = 0; col < picture.width(); col++) {
                // Get the color of each pixel and extract rgb values.
                Color color = picture.get(col, row);
                fV[index] = color.getRed();
                index++;
            }
        }
        return fV;
    }

    // Trains the perceptron on the given training data file.
    public void trainClassifier(String trainFile) {
        // Reads the training file data and extract the features for each image.
        In in = new In(trainFile);
        while (!(in.isEmpty())) {
            Picture picture = new Picture(in.readString());
            double[] features = extractFeatures(picture);
            // Trains the classifier using the corresponding label.
            imageClassifier.trainMulti(features, in.readInt());
        }
    }

    // Returns the name of the class for the given class label.
    public String classNameOf(int classLabel) {
        if (classLabel < 0 || classLabel > (m - 1)) {
            throw new IllegalArgumentException("Class label must be within the "
                                                       + "number of classes.");
        }
        return classNames[classLabel];
    }

    // Returns the predicted class for the given picture.
    public int classifyImage(Picture picture) {
        // For the given image, extract its features and use the
        // multi-perceptron to predict its class label.
        double[] features = extractFeatures(picture);
        return imageClassifier.predictMulti(features);

    }

    // Returns the error rate on the given testing data file.
    // Also prints the misclassified examples - see below.
    public double testClassifier(String testFile) {
        In in = new In(testFile);
        int sum = 0;
        int error = 0;

        // Reads the test file data.
        while (!(in.isEmpty())) {
            sum++;
            String eachImage = in.readString();
            Picture picture = new Picture(eachImage);
            // For each testing image, predicts its class.
            int prediction = classifyImage(picture);
            int classLabel = in.readInt();

            // Prints each misclassified image to standard output and compute
            // the error rate on these images.
            if (prediction != classLabel) {
                error++;
                String correctLabel = "label = " + classNameOf(classLabel) + ", ";
                String misclassified = "predict = " + classNameOf(prediction);
                StdOut.println(eachImage + ", " + correctLabel + misclassified);
            }
        }

        // System.out.println(sum); // Check for problem with adding
        // System.out.println(error); // Check for problem with adding
        return (double) error / sum;
    }

    // Tests this class using a configuration file, training file and test file.
    // See below.
    public static void main(String[] args) {
        ImageClassifier classifier = new ImageClassifier(args[0]);
        classifier.trainClassifier(args[1]);

        double testErrorRate = classifier.testClassifier(args[2]);
        System.out.println("test error rate = " + testErrorRate);
    }
}
