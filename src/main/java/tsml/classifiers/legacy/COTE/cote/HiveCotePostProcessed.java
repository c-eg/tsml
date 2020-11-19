/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
package tsml.classifiers.legacy.COTE.cote;

import experiments.data.DatasetLists;
import java.util.ArrayList;

/**
 *
 * @author Jason Lines (j.lines@uea.ac.uk)
 */
public class HiveCotePostProcessed extends AbstractPostProcessedCote{
    
    private double alpha = 1;
    private boolean useVoting = false;
    
    {
        HiveCotePostProcessed.CLASSIFIER_NAME = "HIVE-COTE";
    }
    public HiveCotePostProcessed(String resultsDir, String datasetName, int resampleId, ArrayList<String> classifierNames) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifierNames = classifierNames;
    }
    
    public HiveCotePostProcessed(String resultsDir, String datasetName, ArrayList<String> classifierNames) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = 0;
        this.classifierNames = classifierNames;
    }
    
    public HiveCotePostProcessed(String resultsDir, String datasetName, int resampleId) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = resampleId;
        this.classifierNames = getDefaultClassifierNames();
    }
    
    public HiveCotePostProcessed(String resultsDir, String datasetName) {
        this.resultsDir = resultsDir;
        this.datasetName = datasetName;
        this.resampleId = 0;
        this.classifierNames = getDefaultClassifierNames();
    }

    public void setAlpha(double alpha){
        this.alpha = alpha;
    }
    
    private void useVotes(){
        this.useVoting = true;
    }
    private void useProbs(){
        this.useVoting = false;
    }
    private ArrayList<String> getDefaultClassifierNames(){
        ArrayList<String> names = new ArrayList<>();
        names.add("EE");
        names.add("ST");
        names.add("RISE");
        names.add("BOSS");
        names.add("TSF");
        return names;
    }
    
    @Override
    public double[] distributionForInstance(int testInstanceId) throws Exception{
        if(useVoting){
            return this.distributionForInstanceWithVoting(testInstanceId);
        }else{
            return this.distributionForInstanceWithProbs(testInstanceId);
        }
    }
    
    public double[] distributionForInstanceWithProbs(int testInstanceId) throws Exception{
        if(this.testDists==null){
            throw new Exception("Error: classifier not initialised correctly. Load results before classifiying.");
        }
        
        int numClasses = this.testDists[0][0].length;
        double[] outDist = new double[numClasses];
        double cvAccSum = 0;
        
        for(int classifier = 0; classifier < testDists.length; classifier++){
            for(int classVal = 0; classVal < numClasses; classVal++){
                outDist[classVal]+= testDists[classifier][testInstanceId][classVal]*(Math.pow(this.cvAccs[classifier],alpha));
            }
            cvAccSum+=(Math.pow(this.cvAccs[classifier],alpha));
        }
        
        for(int classVal = 0; classVal < numClasses; classVal++){
            outDist[classVal]/= cvAccSum;
        }
        
        return outDist;
    }

    public double[] distributionForInstanceWithVoting(int testInstanceId) throws Exception{
        
        if(this.testDists==null){
            throw new Exception("Error: classifier not initialised correctly. Load results before classifiying.");
        }
        
        int numClasses = this.testDists[0][0].length;
        double[] outDist = new double[numClasses];
        double cvAccSum = 0;
        
        int maxId;
        double bsfWeight;
        for(int classifier = 0; classifier < testDists.length; classifier++){
            // find max class value
            maxId = -1;
            bsfWeight = -1;
            for(int classVal = 0; classVal < numClasses; classVal++){
                if(testDists[classifier][testInstanceId][classVal] > bsfWeight){
                    maxId = classVal;
                    bsfWeight = testDists[classifier][testInstanceId][classVal];
                } 
            }
            outDist[maxId]+=(Math.pow(this.cvAccs[classifier],alpha));
            cvAccSum+=(Math.pow(this.cvAccs[classifier],alpha));
        }
        
        for(int classVal = 0; classVal < numClasses; classVal++){
            outDist[classVal]/= cvAccSum;
        }
        
        return outDist;
    }
    
    public static void main(String[] args) throws Exception{
//        String datasetName = "ItalyPowerDemand";
//        Instances train = loadData("C:/users/sjx07ngu/dropbox/tsc problems/"+datasetName+"/"+datasetName+"_TRAIN");
      
//        Instances test = loadData("C:/users/sjx07ngu/dropbox/tsc problems/"+datasetName+"/"+datasetName+"_TEST");      
        /*
            Step 1: build Hive and write to file`
        */
//        
//        HiveCote hc = new HiveCote();
//        hc.makeShouty();
//        hc.turnOnFileWriting("hiveWritingProto/", datasetName);
//        hc.buildClassifier(train);
//        hc.writeTestPredictionsToFile(test, "hiveWritingProto/", datasetName);
        
        /*
            Step 2: read from file and (hhopefully) recreate the same results
        */
        
//        HiveCotePostProcessed hcpp = new HiveCotePostProcessed("hiveWritingProto/", datasetName);
//        hcpp.writeTestSheet("hiveWritingProtoRewrite/");
        
        
        // with alpha = 1 and =4
        HiveCotePostProcessed hcpp;
        double[] alphas = {1.0,4.0};
//        double[] alphas = {1.0};
        ArrayList<String> classifiersToUse = new ArrayList<>();
        classifiersToUse.add("EE_proto");
        classifiersToUse.add("ST_HiveProto");
        classifiersToUse.add("RISE");
        classifiersToUse.add("BOSS");
        classifiersToUse.add("TSF");
        System.out.println("votes");
        for(double alpha:alphas){
            for(String datasetName: DatasetLists.tscProblems85){
                System.out.println(datasetName+" "+alpha);
                for(int resample = 0; resample < 100; resample++){
                    try{
//                        hcpp = new HiveCotePostProcessed("//cmptscsvr.cmp.uea.ac.uk/ueatsc/Results/JayMovingInProgress/", datasetName, resample, classifiersToUse);
//                        hcpp = new HiveCotePostProcessed("C:/3xsshare/Jay/LocalWork/coteConstituentResultsgress/", datasetName, resample, classifiersToUse);
                        hcpp = new HiveCotePostProcessed("C:/3xsshare/Jay/LocalWork/coteConstituentResults/", datasetName, resample, classifiersToUse);
                        hcpp.setAlpha(alpha);
                        hcpp.useVotes();
//                        hcpp.writeTestSheet("hiveWritingProtoRewrite_alpha"+alpha+"/");
                        hcpp.writeTestSheet("hiveWritingProtoRewrite_alpha"+alpha+"_votes/");
//                        hcpp.writeTestSheet("hiveWritingProtoRewrite_alpha"+alpha+"_probs/");
                    }catch(Exception e){
                        System.err.println(datasetName+"_"+resample+"_"+alpha);
//                        e.printStackTrace();
                    }
                }
            }
        }
    }

}
