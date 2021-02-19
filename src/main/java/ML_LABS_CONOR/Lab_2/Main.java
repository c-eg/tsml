package ML_LABS_CONOR.Lab_2;

import ML_LABS_CONOR.Lab_1.HistogramClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
//        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Heights_TRAIN.arff";
//        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Heights_TEST.arff";
//
//        Instances train = WekaTools.loadClassificationData(trainData);
//        Instances test = WekaTools.loadClassificationData(testData);
//
//        // HISTOGRAM CLASSIFIER
//        System.out.println("HISTOGRAM CLASSIFIER");
//        HistogramClassifier hc = new HistogramClassifier();
//        hc.buildClassifier(train);
//
//        int[] predictions = WekaTools.classifyInstances(hc, train);
//        int[] actual = WekaTools.getClassValues(train);
//
//        int[][] confusionMatrix = WekaTools.confusionMatrix(actual, predictions);
//        System.out.println(WekaTools.confusionMatrixToString(confusionMatrix));
//
//        System.out.println("Accuracy: " + WekaTools.accuracy(hc, test));
//        System.out.println("\n");
//
//        // NAIVE BAYES CLASSIFIER
//        System.out.println("NAIVE BAYES CLASSIFIER");
//        NaiveBayes nb = new NaiveBayes();
//        nb.buildClassifier(train);
//
//        int[] predictionsNb = WekaTools.classifyInstances(nb, train);
//        int[] actualNb = WekaTools.getClassValues(train);
//
//        int[][] confusionMatrixNb = WekaTools.confusionMatrix(actualNb, predictionsNb);
//        System.out.println(WekaTools.confusionMatrixToString(confusionMatrixNb));
//
//        System.out.println("Accuracy: " + WekaTools.accuracy(nb, test));
//        System.out.println("\n");
//
//        // MAJORITY CLASS CLASSIFIER
//        System.out.println("MAJORITY CLASS CLASSIFIER");
//        MajorityClassClassifier mcc = new MajorityClassClassifier();
//        mcc.buildClassifier(train);
//
//        int[] predictionsMcc = WekaTools.classifyInstances(mcc, train);
//        int[] actualMcc = WekaTools.getClassValues(train);
//
//        int[][] confusionMatrixMcc = WekaTools.confusionMatrix(actualMcc, predictionsMcc);
//        System.out.println(WekaTools.confusionMatrixToString(confusionMatrixMcc));
//
//        System.out.println("Accuracy: " + WekaTools.accuracy(mcc, test));
//        System.out.println("\n");
//
//        // ZeroR CLASSIFIER
//        System.out.println("ZeroR CLASSIFIER");
//        ZeroR zeroR = new ZeroR();
//        zeroR.buildClassifier(train);
//
//        int[] predictionsR = WekaTools.classifyInstances(zeroR, train);
//        int[] actualR = WekaTools.getClassValues(train);
//
//        int[][] confusionMatrixR = WekaTools.confusionMatrix(actualR, predictionsR);
//        System.out.println(WekaTools.confusionMatrixToString(confusionMatrixR));
//
//        System.out.println("Accuracy: " + WekaTools.accuracy(zeroR, test));
//        System.out.println("\n");

        /*
         * Task 5
         */
        String data = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_2\\Aedes_Female_VS_House_Fly_POWER.arff";

        Instances allData = WekaTools.loadClassificationData(data);


        // a
//        Instances[] split = WekaTools.splitData(allData, 0.3);
//
//        Instances train = split[0];
//        Instances test = split[1];
//
//        ArrayList<Classifier> classifiers = new ArrayList<>();
//
//        IB1 IB1 = new IB1();
//        IBk IBk = new IBk();
//        J48 J48 = new J48();
//        Logistic logistic = new Logistic();
//        HistogramClassifier hc = new HistogramClassifier();
//
//        classifiers.add(IB1);
//        classifiers.add(IBk);
//        classifiers.add(J48);
//        classifiers.add(logistic);
//        classifiers.add(hc);
//
//        for (Classifier c : classifiers) {
//            System.out.println(c.getClass().getSimpleName());
//            c.buildClassifier(train);
//            System.out.println("Accuracy: " + WekaTools.accuracy(c, test) * 100 + "%");
//            System.out.println();
//        }


        // b
        ArrayList<Classifier> classifiers = new ArrayList<>();

        IB1 IB1 = new IB1();
        IBk IBk = new IBk();
        J48 J48 = new J48();
        Logistic logistic = new Logistic();
        HistogramClassifier hc = new HistogramClassifier();

        classifiers.add(IB1);
        classifiers.add(IBk);
        classifiers.add(J48);
        classifiers.add(logistic);
        classifiers.add(hc);

        int classifier = 0;
        int runs = 30;
        double[][] accuracies = new double[classifiers.size()][runs];

        for (Classifier c : classifiers) {
            for (int i = 0; i < runs; i++) {
                Instances[] split = WekaTools.splitData(allData, 0.3);

                Instances train = split[0];
                Instances test = split[1];

                c.buildClassifier(train);
                accuracies[classifier][i] = WekaTools.accuracy(c, test);
            }

            classifier++;
        }

        for (int i = 0; i < classifiers.size(); i++) {
            System.out.println(classifiers.get(i).getClass().getSimpleName() + ": " + Arrays.toString(accuracies[i]));
        }
    }
}
