/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package timeseriesweka.classifiers.proximityForest;

import core.AppContext;
import core.contracts.Dataset;
import datasets.ListDataset;
import evaluation.storage.ClassifierResults;
import experiments.ClassifierLists;
import experiments.Experiments;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import trees.ProximityForest;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * An in-progress wrapper/conversion class for the java Proximity Forest implementation.
 * 
 * The code as-is on the github page does not return distributions when predicting. Therefore
 * in our version of the jar I have added a predict_proba() to be used here instead of predict().
 * Existing proximity code is UNEDITED, and predict_proba simply returns the distribution of 
 * the num_votes class member instead of resolving ties internally and returning a single 
 * class value. NOTE: we are by-passing the test method which is ultimately foreach inst { predict() },
 * and so proximity forest's internal results object is empty. This has no other sides effects 
 * for our own intention, however
 * 
 * 
 * Github code:   https://github.com/fpetitjean/ProximityForestWeka
 * 
 * @article{DBLP:journals/corr/abs-1808-10594,
 *   author    = {Benjamin Lucas and
 *                Ahmed Shifaz and
 *                Charlotte Pelletier and
 *                Lachlan O'Neill and
 *                Nayyar A. Zaidi and
 *                Bart Goethals and
 *                Fran{\c{c}}ois Petitjean and
 *                Geoffrey I. Webb},
 *   title     = {Proximity Forest: An effective and scalable distance-based classifier
 *                for time series},
 *   journal   = {CoRR},
 *   volume    = {abs/1808.10594},
 *   year      = {2018},
 *   url       = {http://arxiv.org/abs/1808.10594},
 *   archivePrefix = {arXiv},
 *   eprint    = {1808.10594},
 *   timestamp = {Mon, 03 Sep 2018 13:36:40 +0200},
 *   biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1808-10594},
 *   bibsource = {dblp computer science bibliography, https://dblp.org}
 * }
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ProximityForestWeka extends AbstractClassifier {

    //taken from appcontext, actual tunable parameters
    private int num_trees = 1;                  //500? look at paper 
    private int num_candidates_per_split = 1;   //5
    private boolean random_dm_per_node = true;  //leave as true
    
    private int numClasses;
    public ProximityForest pf;
    
    public ProximityForestWeka() { 
        pf = new ProximityForest(0);
    }

    public int getNum_trees() {
        return num_trees;
    }

    public void setNum_trees(int num_trees) {
        this.num_trees = num_trees;
    }

    public int getNum_candidates_per_split() {
        return num_candidates_per_split;
    }

    public void setNum_candidates_per_split(int num_candidates_per_split) {
        this.num_candidates_per_split = num_candidates_per_split;
    }

    public boolean isRandom_dm_per_node() {
        return random_dm_per_node;
    }

    public void setRandom_dm_per_node(boolean random_dm_per_node) {
        this.random_dm_per_node = random_dm_per_node;
    }
        
    public void setSeed(int seed) { 
        AppContext.rand_seed = seed;
        AppContext.rand = new Random(seed);        
    }
    
    public int getSeed() { 
        return (int)AppContext.rand_seed;
    }
    
    private Dataset toPFDataset(Instances insts) {
        Dataset dset = new ListDataset(insts.numInstances());
        
        for (Instance inst : insts)
            dset.add((int)inst.classValue(), getSeries(inst));
        
        return dset;
    }
    
    private double[] getSeries(Instance inst) {
        double[] d = new double[inst.numAttributes()-1];
        for (int i = 0; i < d.length; i++)
            d[i] = inst.value(i);
        return d;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        numClasses = data.numClasses();
        Dataset pfdata = toPFDataset(data);
        pf.train(pfdata);
    }
    
    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        return pf.predict_proba(getSeries(inst), numClasses);
    }
    
    
    
    public static void main(String[] args) throws Exception {
        Experiments.ExperimentalArguments exp = new Experiments.ExperimentalArguments();

        exp.dataReadLocation = "Z:\\Data\\TSCProblems2018\\";
        exp.resultsWriteLocation = "C:\\Temp\\ProximityForestWekaTest\\";
        exp.classifierName = "ProximityForest";
        exp.datasetName = "BeetleFly";
        exp.foldId = 0;
        
        Experiments.setupAndRunExperiment(exp);
    }
}
