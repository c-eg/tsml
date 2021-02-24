package ML_LABS_CONOR.Lab_3;

import ML_LABS_CONOR.Lab_2.WekaTools;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instances;

public class Main {
    private static final String DATA = "D:\\Documents\\git\\tsml\\src\\main\\java\\ML_LABS_CONOR\\Lab_3\\breast_cancer.arff";

    public static void main(String[] args) throws Exception {
        //part1();
        //part2();
        part3();
    }

    private static void part3() throws Exception {
        Instances ins = WekaTools.loadClassificationData(DATA);

        Distribution bags = new Distribution(ins);
        InfoGainSplitCrit infoGain = new InfoGainSplitCrit();
        System.out.println(infoGain.splitCritValue(bags));
    }

    private static void part1() throws Exception {
        Instances ins = WekaTools.loadClassificationData(DATA);

        J48 c = new J48();
        c.buildClassifier(ins);
        System.out.println(c);

        J48 c1 = new J48();
        c1.setBinarySplits(true);
        c1.buildClassifier(ins);
        System.out.println(c1);

        J48 c2 = new J48();
        c2.setReducedErrorPruning(true);
        c2.buildClassifier(ins);
        System.out.println(c2);
    }

    private static void part2() throws Exception {
        Instances ins = WekaTools.loadClassificationData(DATA);

        J48 c = new J48();
        System.out.println(c.getCapabilities());
    }
}
