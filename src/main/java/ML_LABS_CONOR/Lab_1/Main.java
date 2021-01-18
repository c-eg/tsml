package ML_LABS_CONOR.Lab_1;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;

import java.io.FileReader;
import java.util.Arrays;


public class Main {
    public static void main(String[] args) throws Exception {
        //partFour();
        partFiveNaiveBayes();
        partFiveiBk();
    }

    private static void partFiveNaiveBayes() throws Exception {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TRAIN.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TEST.arff";

        Instances train = loadData(trainData);
        Instances test = loadData(testData);

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(train.numAttributes() - 1);

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        System.out.println(nb.toString());

        double[] results = test.attributeToDoubleArray(3);
        double countCorrectPredictions = 0;

        for (int i = 0; i < test.numInstances(); i++) {
            Instance t = test.instance(i);
            double prediction = nb.classifyInstance(t);
            if (prediction == results[i])
                countCorrectPredictions++;
        }

        System.out.println("NaiveBayes:");
        System.out.println(countCorrectPredictions + " / " + test.numInstances() + " correct predictions");
        System.out.println((countCorrectPredictions / test.numInstances()) * 100 + "% accuracy");

        for (Instance i : test) {
            System.out.println(Arrays.toString(nb.distributionForInstance(i)));
        }
        System.out.println();
    }

    private static void partFiveiBk() throws Exception {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TRAIN.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TEST.arff";

        Instances train = loadData(trainData);
        Instances test = loadData(testData);

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(train.numAttributes() - 1);

        IBk ibk = new IBk();
        ibk.buildClassifier(train);

        double[] results = test.attributeToDoubleArray(3);
        double countCorrectPredictions = 0;

        for (int i = 0; i < test.numInstances(); i++) {
            Instance t = test.instance(i);
            double prediction = ibk.classifyInstance(t);
            if (prediction == results[i])
                countCorrectPredictions++;
        }

        System.out.println("iBk:");
        System.out.println(countCorrectPredictions + " / " + test.numInstances() + " correct predictions");
        System.out.println((countCorrectPredictions / test.numInstances()) * 100 + "% accuracy");

        for (Instance i : test) {
            System.out.println(Arrays.toString(ibk.distributionForInstance(i)));
        }
        System.out.println();
    }

    private static void partFour() {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TRAIN.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TEST.arff";

        Instances train = loadData(trainData);
        Instances test = loadData(testData);

        System.out.println("Number of instances in train data:\t" + train.numInstances()); // part 4, 1

        System.out.println("Number of attributes in test data:\t" + test.numAttributes()); // part 4, 2

        /*
         * part 4, 3
         */
        int winsCount = 0;

        // loop through the 4th attribute in all instances
        for (double d : train.attributeToDoubleArray(3)) {
            if (d == 2.0) {
                winsCount++;
            }
        }

        System.out.println("Number of wins in train data:\t\t" + winsCount);
        /*
         * end of part 4, 3
         */

        System.out.println("5th instance of test data:\t\t\t" + Arrays.toString(test.get(4).toDoubleArray())); // part 4, 4

        // part 4, 5
        for (Instance i : train) {
            System.out.println(i);
        }

        // part 4, 6
        train.deleteAttributeAt(2);
        test.deleteAttributeAt(2);
        System.out.println("Training instances:\n" + train.toString() + "\n"); // part 4, 6

        train = loadData(trainData);
    }

    public static Instances loadData(String dataLocation) {
        Instances train;

        try {
            FileReader reader = new FileReader(dataLocation);
            train = new Instances(reader);
            return train;
        }
        catch (Exception e) {
            System.out.println("Exception caught: " + e);
        }

        return null;
    }
}
