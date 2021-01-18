package ML_LABS_CONOR.Lab_1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.stream.IntStream;

public class HistogramClassifier implements Classifier
{
    private int bins = 10;
    private int classIndex = 0;
    private double max = 0;
    private double min = 0;
    double range;
    double intervals;

    // histograms
    private int[][] histograms;

    public int getBins() {
        return bins;
    }

    public void setBins(int bins) {
        this.bins = bins;
    }

    public int getClassIndex() {
        return classIndex;
    }

    public void setClassIndex(int classIndex) {
        this.classIndex = classIndex;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception
    {
        histograms = new int[instances.numClasses()][bins];

        // set min and max
        for (Instance ins : instances) {
            double val = ins.value(0);

            if (min == 0) {
                min = val;
                max = val;
            }
            else if (val < min)
                min = val;
            else if (val > max)
                max = val;
        }

        // set range of data and interval size for each bin
        range = max - min;
        intervals = range / bins;

        // for each class
        for (int i = 0; i < instances.numClasses(); i++) {
            for (Instance ins : instances) {
                if (ins.classValue() == i) {
                    double val = ins.value(0);

                    // fill histogram with counters for each bin
                    for (int j = 0; j < bins; j++) {
                        // if val is in interval range, add 1 to counter
                        if (val >= min + intervals * j && val <= min + intervals * (j + 1)) {
                            histograms[i][j] += 1;
                        }
                    }
                }
            }
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception
    {
        double[] d = distributionForInstance(instance); // get probability for both classes
        return getHighestIndex(d); // return index of highest probability
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception
    {
        double[] d = new double[instance.numClasses()];
        double val = instance.value(0);
        int index = 0;

        // find correct interval index from instance passed
        for (int i = 0; i < bins; i++) {
            if (val >= (min + intervals * i) && val <= (min + intervals * (i+1))) {
                index = i;
                break;
            }
        }

        // work out relative frequency for each histogram for that interval
        for (int i = 0; i < histograms.length; i++) {
            int total = IntStream.of(histograms[i]).sum(); // total values in each histogram
            double rFreq = histograms[i][index] / (double)total; // relative frequency for interval
            d[i] = rFreq; // set relative frequency to class index
        }

        return d;
    }

    private int getHighestIndex(double[] data) {
        int largest = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > data[largest])
                largest = i;
        }

        return largest;
    }

    @Override
    public Capabilities getCapabilities()
    {
        return null;
    }

    public static void main(String[] args) throws Exception {
        String trainData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\weight-height.arff";
        String testData = "D:\\OneDrive\\OneDrive - University of East Anglia\\Uni Work\\Year 3\\Machine Learning\\Labs\\Lab 1\\heights_TEST.arff";

        Instances train = Main.loadData(trainData);
        Instances test = Main.loadData(testData);

        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(train.numAttributes() - 1);

        HistogramClassifier hc = new HistogramClassifier();
        hc.buildClassifier(train);

        double[] results = test.attributeToDoubleArray(1);
        double countCorrectPredictions = 0;

        for (int i = 0; i < test.numInstances(); i++)
        {
            Instance t = test.instance(i);
            double prediction = hc.classifyInstance(t);
            if (prediction == results[i])
                countCorrectPredictions++;
        }

        System.out.println("hc:");
        System.out.println(countCorrectPredictions + " / " + test.numInstances() + " correct predictions");
        System.out.println(countCorrectPredictions/test.numInstances() + "% accuracy");

        for (Instance i : test)
        {
            System.out.println(Arrays.toString(hc.distributionForInstance(i)));
        }
        System.out.println();
    }
}
