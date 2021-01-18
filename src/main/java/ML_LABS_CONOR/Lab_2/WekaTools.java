package ML_LABS_CONOR.Lab_2;

import weka.classifiers.Classifier;
import weka.core.*;

import java.io.FileReader;
import java.util.Random;

public class WekaTools {
    /**
     * Function to find accuracy of classifier, given test data.
     * @param c classifier
     * @param test instances
     * @return double correct predictions out of total instances
     */
    public static double accuracy(Classifier c, Instances test) throws Exception {
        double countCorrectPredictions = 0;
        double[] resultsFromData = test.attributeToDoubleArray(test.numAttributes() - 1);

        for (int i = 0; i < test.numInstances(); i++)
        {
            Instance t = test.instance(i);
            double prediction = c.classifyInstance(t);

            if (prediction == resultsFromData[i])
                countCorrectPredictions++;
        }

        return countCorrectPredictions;
    }

    /**
     * Function to get Instances from data file passed.
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
            System.out.println("Exception caught: " + e);
        }

        return null;
    }

    /**
     * Splits the data into train and test splits. Also randomises instance order
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
     * @param predicted predicted results
     * @param actual actual results
     * @return Contingency Table/Confusion Matrix
     * <p>
     *     e.g.
     *     predicted = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
     *     actual =    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
     *
     *     Returns:
     *     [1, 0]
     *     [3, 6]
     *
     *     Left column is class actual 0, right is 1
     *     First row is class predicted 0, 2nd is 1
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
}
