package ML_LABS_CONOR.Lab_2;

import ML_LABS_CONOR.Lab_1.HistogramClassifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Heights_TRAIN.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Heights_TEST.arff";

        Instances train = WekaTools.loadClassificationData(trainData);
        Instances test = WekaTools.loadClassificationData(testData);

        // HISTOGRAM CLASSIFIER
        System.out.println("HISTOGRAM CLASSIFIER");
        HistogramClassifier hc = new HistogramClassifier();
        hc.buildClassifier(train);

        int[] predictions = WekaTools.classifyInstances(hc, train);
        int[] actual = WekaTools.getClassValues(train);

        int[][] confusionMatrix = WekaTools.confusionMatrix(actual, predictions);
        System.out.println(WekaTools.confusionMatrixToString(confusionMatrix));

        System.out.println("Accuracy: " + WekaTools.accuracy(hc, test));

        System.out.println("\n");

        // NAIVE BAYES CLASSIFIER
        System.out.println("NAIVE BAYES CLASSIFIER");
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        int[] predictionsNb = WekaTools.classifyInstances(nb, train);
        int[] actualNb = WekaTools.getClassValues(train);

        int[][] confusionMatrixNb = WekaTools.confusionMatrix(actualNb, predictionsNb);
        System.out.println(WekaTools.confusionMatrixToString(confusionMatrixNb));

        System.out.println("Accuracy: " + WekaTools.accuracy(nb, test));
    }
}
