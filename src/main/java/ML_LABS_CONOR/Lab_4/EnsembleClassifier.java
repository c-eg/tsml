package ML_LABS_CONOR.Lab_4;

import ML_LABS_CONOR.Lab_2.WekaTools;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

public class EnsembleClassifier implements Classifier {
    private int numOfTrees = 10;
    private J48[] trees;

    public void setNumOfTrees(int numOfTrees) {
        this.numOfTrees = numOfTrees;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        trees = new J48[numOfTrees];
        int attributesPerTree = data.numAttributes()/numOfTrees;

        Random r = new Random();

        // loop through number of trees
        for (int i = 0; i < numOfTrees; i++) {
            // create temp instances and fill with random data
            Instances temp = new Instances(data, 0);

            for (int j = 0; j < attributesPerTree; j++) {
                int index = r.nextInt(data.size()); // get random index
                temp.add(data.get(index)); // add instance at index to temp
                data.remove(index); // remove instance and index
            }

            // build classifier on temp data
            trees[i] = new J48();
            trees[i].buildClassifier(temp);
        }

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probabilities = distributionForInstance(instance);
        return WekaTools.getHighestIndex(probabilities);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int[] votes = new int[instance.numClasses()];
        double[] probabilities = new double[instance.numClasses()];

        // for each tree
        for (J48 tree : trees) {
            // get the prediction index from J48 classifier
            double indexPrediction = tree.classifyInstance(instance);

            // increment votes at index of prediction
            votes[(int)indexPrediction]++;
        }

        // get probabilities for each class
        for (int i = 0; i < probabilities.length; i++) {
            /*
             * If votes for class 0 were 2. And num trees were 10.
             * probability for class 0 would be 2/10.
             */
            probabilities[i] = (double) votes[i] / numOfTrees;
        }

        // return the index of the highest number of votes
        return probabilities;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
