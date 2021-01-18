package ML_LABS_CONOR.Lab_2;

import weka.core.Instances;

public class Main {
    public static void main(String[] args) {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TRAIN.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TEST.arff";

        Instances train = WekaTools.loadClassificationData(trainData);
        Instances test = WekaTools.loadClassificationData(testData);

        System.out.println(train + "\n\n");
        Instances[] something = WekaTools.splitData(train, 0.5);
        System.out.println(something[0] + "\n\n");
        System.out.println(something[1] + "\n\n");
    }
}
