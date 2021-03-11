package ML_LABS_CONOR.Lab_6;

import ML_LABS_CONOR.Lab_2.WekaTools;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

public class kNN extends AbstractClassifier {
    private Instances data;
    private int kNeighbours;

    public kNN() {
        this.kNeighbours = 1;
    }

    public kNN(int kNeighbours) {
        this.kNeighbours = kNeighbours;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.data = data;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probabilities = distributionForInstance(instance);
        return WekaTools.getHighestIndex(probabilities);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        // create double array of all distances
        ArrayList<InstanceDistancePair> distancePairs = new ArrayList<>();

        // loop through each instance and calculate distance
        for (Instance ins : data) {
            double distance = distance(instance, ins);

            // add each instance and distance to an object to allow sorting
            InstanceDistancePair pair = new InstanceDistancePair(ins, distance);
            distancePairs.add(pair);
        }

        // sort array by closest distance first
        Collections.sort(distancePairs);

        // calculate the most common index in the closest k neighbours
        double[] classLikelihoods = new double[kNeighbours];
        for (int i = 0; i < kNeighbours; i++) {
            classLikelihoods[(int) distancePairs.get(i).instance.classValue()]++;
        }

        for (int i = 0; i < kNeighbours; i++) {
            classLikelihoods[i] = classLikelihoods[i] / kNeighbours;
        }

        return classLikelihoods;
    }

    public void setkNeighbours(int kNeighbours) {
        this.kNeighbours = kNeighbours;
    }

    /**
     * Gets the Euclidean distance between two instance objects.
     *
     * @param x data
     * @param y data
     * @return Euclidean distance
     */
    double distance(Instance x, Instance y) {
        double distance = 0;

        double[] xData = x.toDoubleArray();
        double[] yData = y.toDoubleArray();

        // .length - 1 because don't want to include class value in calculation
        for (int i = 0; i < xData.length - 1; i++) {
            distance += Math.pow((xData[i] - yData[i]), 2);
        }

        return distance;
        //return Math.sqrt(distance);
    }

    private static class InstanceDistancePair implements Comparable<InstanceDistancePair> {
        private final Instance instance;
        private final double distance;

        public InstanceDistancePair(Instance instance, double distance) {
            this.instance = instance;
            this.distance = distance;
        }

        @Override
        public int compareTo(InstanceDistancePair other) {
            return Double.compare(this.distance, other.distance);
        }

        @Override
        public String toString() {
            return String.valueOf(this.distance);
        }
    }
}
