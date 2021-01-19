package ML_LABS_CONOR.Lab_2;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class MajorityClassClassifier implements Classifier {
    private double mostOccurringClass;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // store most commonly occurring class
        double[] classDistribution = WekaTools.classDistribution(data);
        mostOccurringClass = WekaTools.getHighestIndex(classDistribution);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return mostOccurringClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
