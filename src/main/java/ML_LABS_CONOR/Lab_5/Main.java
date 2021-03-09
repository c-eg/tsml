package ML_LABS_CONOR.Lab_5;

import ML_LABS_CONOR.Lab_1.HistogramClassifier;
import ML_LABS_CONOR.Lab_2.WekaTools;
import evaluation.storage.ClassifierResults;
import fileIO.OutFile;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        //part1_1();
        //part1_2();
        part2();
    }

    private static void part1_1() throws Exception {
        // set path locations
        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_5\\";
        String dataset = "JW_RedVsBlack";

        // create train and test instances
        Instances train = WekaTools.loadClassificationData(path + dataset + "_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData(path + dataset + "_TEST.arff");

        // create and build classifier using train data
        NaiveBayes classifier = new NaiveBayes();
        WekaTools.generateTestResults(classifier, train, test, path, "TestOutput");
    }

    private static void part1_2() throws Exception {
        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_5\\";
        String testOutput = "TestOutput.csv";

        ClassifierResults res = new ClassifierResults();
        res.loadResultsFromFile(path + testOutput);
        res.findAllStats();

        System.out.println("Accuracy: " + res.getAcc());
        System.out.println("Balanced Accuracy: " + res.balancedAcc);
        System.out.println("Negative Log Likelihood: " + res.nll);
        System.out.println("Area Under Curve: " + res.meanAUROC);
    }

    public static void part2() throws Exception {
        // set path locations
        String path = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_5\\";
        String dataset = "JW_RedVsBlack";

        Instances data = WekaTools.loadClassificationData(path + dataset + ".arff");

        NaiveBayes classifier = new NaiveBayes();
        int numFolds = 10;

        // create output file
        OutFile out = new OutFile(path + "TestOutput10Folds.csv");
        out.writeLine("NaiveBayes");
        out.writeLine("No Parameter Info");
        //out.writeLine(String.valueOf(WekaTools.accuracy(classifier, test)));

        for (int currentFold = 0; currentFold < numFolds; currentFold++) {
            Instances train = data.trainCV(numFolds, currentFold);
            Instances test = data.testCV(numFolds, currentFold);

            classifier.buildClassifier(train);

            for (Instance ins : test) {
                int prediction = (int) classifier.classifyInstance(ins);
                double[] probabilities = classifier.distributionForInstance(ins);

                out.writeString((int) ins.classValue() + "," + prediction + ",,");

                StringBuilder line = new StringBuilder();
                for (double d : probabilities) {
                    line.append(d).append(",");
                }
                line.deleteCharAt(line.length() - 1);

                out.writeLine(line.toString());
            }
        }
    }
}
