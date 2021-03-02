package ML_LABS_CONOR.Lab_4;

import ML_LABS_CONOR.Lab_2.WekaTools;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.File;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws Exception {
        //part1();
        part5();
    }

    private static void part1() throws Exception {
        double[] baggingAcc = new double[33];
        double[] randomForestAcc = new double[33];

        String basePath = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_4\\UCIContinuous\\";

        File dataDir = new File(basePath);
        int fileCounter = 0;

        for (File f : dataDir.listFiles()) {
            if (f.isDirectory()) {
                String newPath = basePath + f.getName() + "\\" + f.getName() + ".arff";

                Instances data = WekaTools.loadClassificationData(newPath);
                Instances[] split = WekaTools.splitData(data, 0.3);

                Instances train = split[0];
                Instances test = split[1];

                Bagging b = new Bagging();
                b.buildClassifier(train);
                baggingAcc[fileCounter] = WekaTools.accuracy(b, test);

                RandomForest r = new RandomForest();
                r.buildClassifier(train);
                randomForestAcc[fileCounter] = WekaTools.accuracy(r, test);

                fileCounter++;
            }
        }

        System.out.println("Bagging Accuracies:");
        System.out.println(Arrays.toString(baggingAcc));

        System.out.println("RandomForest Accuracies:");
        System.out.println(Arrays.toString(randomForestAcc));
    }

    private static void part5() throws Exception {
        double[] ensembleAcc = new double[33];

        String basePath = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_4\\UCIContinuous\\";

        File dataDir = new File(basePath);
        int fileCounter = 0;

        for (File f : dataDir.listFiles()) {
            if (f.isDirectory()) {
                String newPath = basePath + f.getName() + "\\" + f.getName() + ".arff";

                Instances data = WekaTools.loadClassificationData(newPath);
                Instances[] split = WekaTools.splitData(data, 0.3);

                Instances train = split[0];
                Instances test = split[1];

                EnsembleClassifier e = new EnsembleClassifier();
                e.buildClassifier(train);
                ensembleAcc[fileCounter] = WekaTools.accuracy(e, test);

                fileCounter++;
            }
        }

        System.out.println("Ensemble Classifier Accuracies:");
        System.out.println(Arrays.toString(ensembleAcc));
    }
}
