package ML_LABS_CONOR.Lab_6;

import ML_LABS_CONOR.Lab_2.WekaTools;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
    public static void main(String[] args) throws Exception {
        part1();
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
}
