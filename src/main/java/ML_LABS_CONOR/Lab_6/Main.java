package ML_LABS_CONOR.Lab_6;

import ML_LABS_CONOR.Lab_2.WekaTools;
import evaluation.storage.ClassifierResults;
import weka.classifiers.lazy.IB1;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        //part1();
        //part2();
        part3();
    }

    private static void part1() throws Exception {
        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_1\\";
        String fileName = "Arsenal";

//        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_4\\UCIContinuous\\blood\\";
//        String fileName = "blood";

        Instances train = WekaTools.loadClassificationData(path + fileName + "_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData(path + fileName + "_TEST.arff");

        OneNN classifier = new OneNN();
        classifier.buildClassifier(train);

        for (Instance ins : test) {
            System.out.print("Actual class: " + ins.classValue() + " | ");
            System.out.println("Predicted class: " + classifier.classifyInstance(ins));
        }
    }

    private static void part2() throws Exception {
        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_6\\";
        String fileName = "FootballPlayers";

        Instances data = WekaTools.loadClassificationData(path + fileName + ".arff");
        Instances[] split = WekaTools.splitData(data, 0.3);

        Instances train = split[0];
        Instances test = split[1];

        /*
         * OneNN
         */
        System.out.println("OneNN Classifier");
        OneNN oneNN = new OneNN();
        oneNN.buildClassifier(train);

        WekaTools.generateTestResults(oneNN, train, test, "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_6\\", "OneNNTestResults");
        String OneNNtestOutput = "OneNNTestResults.csv";

        ClassifierResults oneNNres = new ClassifierResults();
        oneNNres.loadResultsFromFile(path + OneNNtestOutput);
        oneNNres.findAllStats();

        System.out.println("Accuracy: " + oneNNres.getAcc());
        System.out.println("Balanced Accuracy: " + oneNNres.balancedAcc);
        System.out.println("Negative Log Likelihood: " + oneNNres.nll);
        System.out.println("Area Under ROC: " + oneNNres.meanAUROC);

        /*
         * IB1
         */
        System.out.println("\n\nIB1 Classifier");
        IB1 ib1 = new IB1();
        ib1.buildClassifier(train);

        WekaTools.generateTestResults(ib1, train, test, "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_6\\", "IB1TestResults");
        String ib1TestOutput = "IB1TestResults.csv";

        ClassifierResults ib1res = new ClassifierResults();
        ib1res.loadResultsFromFile(path + ib1TestOutput);
        ib1res.findAllStats();

        System.out.println("Accuracy: " + ib1res.getAcc());
        System.out.println("Balanced Accuracy: " + ib1res.balancedAcc);
        System.out.println("Negative Log Likelihood: " + ib1res.nll);
        System.out.println("Area Under ROC: " + ib1res.meanAUROC);
    }

    private static void part3() throws Exception {
//        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_6\\";
//        String fileName = "FootballPlayers";

        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_4\\UCIContinuous\\blood\\";
        String fileName = "blood";

        Instances data = WekaTools.loadClassificationData(path + fileName + ".arff");
        Instances[] split = WekaTools.splitData(data, 0.3);

        Instances train = split[0];
        Instances test = split[1];

        /*
         * kNN
         */
        System.out.println("kNN Classifier");

        kNN kNN = new kNN(20);
        kNN.buildClassifier(train);

        WekaTools.generateTestResults(kNN, train, test, "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_6\\", "kNNTestResults");
        String kNNtestOutput = "kNNTestResults.csv";

        ClassifierResults kNNres = new ClassifierResults();
        kNNres.loadResultsFromFile("D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_6\\" + kNNtestOutput);
        kNNres.findAllStats();

        System.out.println("Accuracy: " + kNNres.getAcc());
        System.out.println("Balanced Accuracy: " + kNNres.balancedAcc);
        System.out.println("Negative Log Likelihood: " + kNNres.nll);
        System.out.println("Area Under ROC: " + kNNres.meanAUROC);
    }
}
