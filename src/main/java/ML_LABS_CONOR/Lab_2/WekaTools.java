package ML_LABS_CONOR.Lab_2;

import weka.classifiers.Classifier;
import weka.core.*;

import java.io.FileReader;
import java.util.Random;

public class WekaTools {
    /**
     * Function to find accuracy of classifier, given test data.
     *
     * @param c classifier
     * @param test instances
     * @return accuracy 5/10 correct would return 0.5
     */
    public static double accuracy(Classifier c, Instances test) {
        double countCorrectPredictions = 0;
        double[] resultsFromData = test.attributeToDoubleArray(test.numAttributes() - 1);

        for (int i = 0; i < test.numInstances(); i++) {
            Instance t = test.instance(i);
            double prediction = 0;
            try {
                prediction = c.classifyInstance(t);
            }
            catch (Exception e) {
                System.err.println("Exception caught: " + e.getMessage());
            }

            if (prediction == resultsFromData[i])
                countCorrectPredictions++;
        }

        return countCorrectPredictions / test.numInstances();
    }

    /**
     * Function to get Instances from data file passed.
     *
     * @param fullPath location string to data
     * @return Instances from data
     */
    public static Instances loadClassificationData(String fullPath) {
        Instances train;

        try {
            FileReader reader = new FileReader(fullPath);
            train = new Instances(reader);
            train.setClassIndex(train.numAttributes() - 1);
            return train;
        }
        catch (Exception e) {
            System.err.println("Exception caught: " + e.getMessage());
        }

        return null;
    }

    /**
     * Splits the data into train and test splits. Also randomises instance
     * order.
     *
     * @param all data
     * @param proportion proportion of data to be split, i.e. '0.5' = 50%
     * @return Instances array of size 2, containing split data
     */
    public static Instances[] splitData(Instances all, double proportion) {
        // resample data to make random
        long time = System.nanoTime();
        Random random = new Random(time);
        Instances resampled = all.resample(random);

        // create splits
        Instances[] split = new Instances[2];
        split[0] = new Instances(resampled);
        split[1] = new Instances(resampled, 0);

        // MOVE a proportion of the data from split[0] to split[1]
        double toMove = proportion * resampled.numInstances();
        for (int i = 0; i < toMove; i++) {
            Instance temp = split[0].remove(0);
            split[1].add(temp);
        }

        return split;
    }

    /**
     * Function to find class distribution for data.
     *
     * e.g. a three class problem - 200: 0; 500: 1; 300: 2.
     * Would return [0.2, 0.5, 0.3]
     * @param data data
     * @return class distribution
     */
    public static double[] classDistribution(Instances data) {
        double[] d = new double[data.numClasses()];

        for (int i = 0; i < data.numInstances(); i++) { // loop for each instance
            for (int j = 0; j < data.numClasses(); j++) { // loop for each class
                if (data.get(i).classValue() == j) {
                    d[j]++;
                    break;
                }
            }
        }

        for (int i = 0; i < d.length; i++) {
            d[i] = d[i] / data.numInstances();
        }

        return d;
    }

    /**
     * Function to generate Contingency Table/Confusion Matrix from a set of
     * correct results and predicted results.
     *
     * @param predicted predicted results
     * @param actual actual results
     * @return Contingency Table/Confusion Matrix
     * <p>
     * e.g.
     * predicted = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
     * actual =    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
     *
     * Returns:
     *            Actual
     * Predicted / 0  1
     *     0      [1, 0]
     *     1      [3, 6]
     * </p>
     */
    public static int[][] confusionMatrix(int[] predicted, int[] actual) {
        int[][] matrix = new int[2][2];

        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == actual[i]) {
                if (predicted[i] == 0)
                    matrix[0][0]++;
                else if (predicted[i] == 1)
                    matrix[1][1]++;
            }
            else {
                if (predicted[i] == 0)
                    matrix[0][1]++;
                else if (predicted[i] == 1)
                    matrix[1][0]++;
            }
        }

        return matrix;
    }

    /**
     * Function to return confusion matrix in an easily-understandable way.
     *
     * @param confusionMatrix confusionMatrix
     * @return confusion matrix is easy-to-read form
     */
    public static String confusionMatrixToString(int[][] confusionMatrix) {
        StringBuilder sb = new StringBuilder();

        sb.append("           Actual\n");
        sb.append("Predicted / 0  1\n");
        sb.append("    0   [").append(confusionMatrix[0][0]).append("  ").append(confusionMatrix[0][1]).append("]").append("\n");
        sb.append("    1   [").append(confusionMatrix[1][0]).append("  ").append(confusionMatrix[1][1]).append("]").append("\n");

        return sb.toString();
    }

    /**
     * Function to get predicted class values from Instances and a built
     * classifier.
     *
     * @param c classifier (already built)
     * @param data Instances
     * @return array of class value predictions
     */
    public static int[] classifyInstances(Classifier c, Instances data) {
        int[] predictedClassValues = new int[data.numInstances()];

        for (int i = 0; i < predictedClassValues.length; i++) {
            try {
                predictedClassValues[i] = (int) c.classifyInstance(data.get(i));
            }
            catch (Exception e) {
                System.err.println("Exception caught: " + e.getMessage());
            }
        }

        return predictedClassValues;
    }

    /**
     * Function to get actual class values from data.
     *
     * @param data data passed
     * @return actual class values
     */
    public static int[] getClassValues(Instances data) {
        int[] classValues = new int[data.numInstances()];

        for (int i = 0; i < classValues.length; i++) {
            try {
                classValues[i] = (int) data.get(i).classValue();
            }
            catch (Exception e) {
                System.err.println("Exception caught: " + e.getMessage());
            }
        }

        return classValues;
    }

    /**
     * Function to get index of highest value in double array.
     *
     * @param data array
     * @return int index of highest value
     */
    public static int getHighestIndex(double[] data) {
        int largest = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > data[largest])
                largest = i;
        }

        return largest;
    }
}
