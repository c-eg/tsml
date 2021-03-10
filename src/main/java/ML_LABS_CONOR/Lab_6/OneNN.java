package ML_LABS_CONOR.Lab_6;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class OneNN extends AbstractClassifier {
    private Instances data;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        this.data = data;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        // get distance for first instance in data
        Instance closest = data.get(0);
        double closestDistance = distance(instance, closest);

        // loop through the rest of instances in data and check if the distance is smaller
        for (int i = 1; i < data.numInstances(); i++) {
            Instance current = data.get(i);
            double distance = distance(instance, current);

            // if current distance is lower than previous ones, set to new closest
            if (distance < closestDistance) {
                closest = current;
                closestDistance = distance;
            }
        }

        return closest.classValue();
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] probabilities = new double[instance.numClasses()];
        double prediction = classifyInstance(instance); // get predicted class

        // set predicted class value to 1 and all others to 0
        for (int i = 0; i < probabilities.length; i++) {
            if (prediction == i)
                probabilities[i] = 1;
            else
                probabilities[i] = 0;
        }

        return probabilities;
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

        return Math.sqrt(distance);
    }
}
