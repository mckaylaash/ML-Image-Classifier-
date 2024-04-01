# ML-Image-Classifier-
Implements the perceptron algorithm to classify images; Applicable to handwriting, animals, characters, etc

The Perceptron class is the starting model for which we base the MultiPerceptron class off of. This class implements several methods, including a training and prediction methods, and a specific toString method to output the array of weighted sums as a vector. The main method makes an individual call to each method and tests each by printing out the weights vector with each round of training. 

Note: The Perceptron class uses the prediction method to test. This is called in the main array.

There are two MultiPerceptron classes included in the repo. One uses an array of Perceptron objects to store multiple perceptrons, and thus is more efficient in terms of the size of the weighted vector. The main method uses several arrays to train the MultiPerceptron, then uses separate ***different*** testing vectors and calls the prediction method. 

The second MultiPerceptron class does not use the Perceptron class and instead creates one large vector to store each weighted vector and increments the index proportional to the size of each training and testing array. 

Note: This class is significantly ***LESS*** efficient and should not be used for extremely large training vectors. 

The Image Classifier class uses a MultiPerceptron to act as an ImageClassifier, and can extract features from grayscale pixel colors of each picture sent in as training or testing data. The main method uses a configuration file to tets this class, and uses separate training data when calling the testClassifier method.
