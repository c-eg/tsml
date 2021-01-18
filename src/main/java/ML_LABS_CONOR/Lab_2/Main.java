package ML_LABS_CONOR.Lab_2;

import weka.core.Instances;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TRAIN.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\Arsenal_TEST.arff";

        Instances train = WekaTools.loadClassificationData(trainData);
        Instances test = WekaTools.loadClassificationData(testData);

        //System.out.println(Arrays.toString(WekaTools.classDistribution(train)) + "\n\n");


        int[] actual = new int[]{0, 0, 1, 1, 1, 0, 0, 1, 1, 1};
        int[] predicted = new int[]{0, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        int[][] confusionMatrix = WekaTools.confusionMatrix(predicted, actual);
        System.out.println(Arrays.deepToString(confusionMatrix));
    }
}
