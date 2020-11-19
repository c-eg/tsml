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
package tsml.transformers;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import experiments.data.DatasetLoading;
import fileIO.OutFile;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <!-- globalinfo-start --> Implementation of power spectrum function as a Weka
 * SimpleBatchFilter Series to series transform independent of class value <!--
 * globalinfo-end --> <!-- options-start --> Valid options are:
 * <p/>
 * TO DO <!-- options-end -->
 *
 * 
 * author: Anthony Bagnall circa 2008. Reviewed and tidied up 2019
 */
public class PowerSpectrum extends FFT {
    boolean log = false;
    FFT fftTransformer;

    public void takeLogs(boolean x) {
        log = x;
    }

    public PowerSpectrum() {
        fftTransformer = new FFT();
        fftTransformer.useDFT();
    }

    public void useFFT() {
        fftTransformer.useFFT();
    }

    @Override
    public Instances determineOutputFormat(Instances inputFormat) {
        // Set up instances size and format.
        int length = (fftTransformer.findLength(inputFormat));
        length /= 2;
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int i = 0; i < length; i++) {
            name = "PowerSpectrum_" + i;
            atts.add(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0) { // Classification set, set class
            // Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            ArrayList<String> vals = new ArrayList(target.numValues());
            for (int i = 0; i < target.numValues(); i++)
                vals.add(target.value(i));
            atts.add(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("PowerSpectrum" + inputFormat.relationName(), atts,
                inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0)
            result.setClassIndex(result.numAttributes() - 1);

        return result;
    }

    @Override
    public Instance transform(Instance inst){
        Instance f=fftTransformer.transform(inst);
        int length = f.numAttributes();
        if (inst.classIndex() >= 0)
            length--;
        length /= 2;

        Instance out=new DenseInstance(length + inst.classIndex() >= 0 ? 1 : 0);

        if(log)
        {
            double l1;		
            for(int j=0;j<length;j++){
                l1= Math.sqrt(f.value(j*2)*f.value(j*2)+f.value(j*2+1)*f.value(j*2+1));
                out.setValue(j,Math.log(l1));
            }
        }
        else{
            for (int j = 0; j < length; j++) {
                out.setValue(j, Math.sqrt(f.value(j * 2) * f.value(j * 2) + f.value(j * 2 + 1) * f.value(j * 2 + 1)));
            }
        }

        //Set class value.
        if(inst.classIndex()>=0)
            out.setValue(length, f.classValue());

        return out;
    }

    @Override
	public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        TimeSeriesInstance f_inst =fftTransformer.transform(inst);

        List<List<Double>> out_data = new ArrayList<>();

        int length = inst.getMaxLength() / 2;
        
        for(TimeSeries f : f_inst){
            List<Double> vals = new ArrayList<>(length);
            if(log){
                double l1;		
                for(int j=0;j<length;j++){
                    l1= Math.sqrt(f.get(j*2)*f.get(j*2)+f.get(j*2+1)*f.get(j*2+1));
                    vals.set(j,Math.log(l1));
                }
            }
            else{
                for (int j = 0; j < length; j++) {
                    vals.set(j, Math.sqrt(f.get(j * 2) * f.get(j * 2) + f.get(j * 2 + 1) * f.get(j * 2 + 1)));
                }
            }

            out_data.add(vals);
        }

        return new TimeSeriesInstance(out_data, inst.getLabelIndex());
    }


    public static void waferTest() {
        /*
         * Instances a=WekaMethods.
         * loadDataThrowable("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TRAIN"
         * ); Instances b=WekaMethods.
         * loadDataThrowable("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TEST"
         * ); PowerSpectrum ps=new PowerSpectrum(); try{ Instances c=ps.process(a);
         * Instances d=ps.process(b); OutFile of = new
         * OutFile("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TRAIN_PS.arff"
         * ); OutFile of2 = new
         * OutFile("C:\\Research\\Data\\Time Series Data\\Time Series Classification\\wafer\\wafer_TEST_PS.arff"
         * ); of.writeString(c.toString()); of2.writeString(d.toString());
         * }catch(Exception e){ System.out.println(" Exception ="+e); }
         */ }

    /* Transform by the built in filter */
    public static double[] powerSpectrum(double[] d) {

        // Check power of 2
        if (((d.length) & (d.length - 1)) != 0) // Not a power of 2
            return null;
        FFT.Complex[] c = new FFT.Complex[d.length];
        for (int j = 0; j < d.length; j++) {
            c[j] = new FFT.Complex(d[j], 0.0);
        }
        FFT f = new FFT();
        f.fft(c, c.length);
        double[] ps = new double[c.length];
        for (int i = 0; i < c.length; i++)
            ps[i] = c[i].getReal() * c[i].getReal() + c[i].getImag() * c[i].getImag();
        return ps;
    }

    public static Instances loadData(String fullPath) {
        Instances d = null;
        FileReader r;
        int nosAtts;
        try {
            r = new FileReader(fullPath + ".arff");
            d = new Instances(r);
            d.setClassIndex(d.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println("Unable to load data on path " + fullPath + " Exception thrown =" + e);
            e.printStackTrace();
            System.exit(0);
        }
        return d;
    }

    public static void matlabComparison() {

        // MATLAB Output generated by
        // Power of 2: use FFT
        // Create set of instances with 16 attributes, with values
        // Case 1: All Zeros
        // Case 2: 1,2,...16
        // Case 3: -8,-7, -6,...,0,1,...7
        // Case 4: 0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1
        /*
         * PowerSpectrum ps=new PowerSpectrum();
         * 
         * Instances test1=ClassifierTools.
         * loadDataThrowable("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\FFT_test1"
         * ); Instances test2=ClassifierTools.
         * loadDataThrowable("C:\\Users\\ajb\\Dropbox\\TSC Problems\\TestData\\FFT_test2"
         * ); Instances t2; try{ t2=ps.process(test1);
         * System.out.println(" TEST 1 PS ="+t2); t2=ps.process(test2);
         * System.out.println(" TEST 2 PS ="+t2);
         * 
         * 
         * }catch(Exception e){ System.out.println(" Errrrrrr = "+e);
         * e.printStackTrace(); System.exit(0); }
         */
        // Not a power of 2: use padding

        // Not a power of 2: use truncate

        // Not a power of 2: use DFT

    }

    public static void main(String[] args) {
        String problemPath = "E:/TSCProblems/";
        String resultsPath = "E:/Temp/";
        String datasetName = "ItalyPowerDemand";
        Instances train = DatasetLoading
                .loadDataNullable("E:/TSCProblems/" + datasetName + "/" + datasetName + "_TRAIN");
        PowerSpectrum ps = new PowerSpectrum();
        try {
            Instances trans = ps.transform(train);
            OutFile out = new OutFile(resultsPath + datasetName + "PS_JAVA.csv");
            out.writeLine(datasetName);
            for (Instance ins : trans) {
                double[] d = ins.toDoubleArray();
                for (int j = 0; j < d.length; j++) {
                    if (j != trans.classIndex())
                        out.writeString(d[j] + ",");
                }
                out.writeString("\n");
            }
        } catch (Exception ex) {
            System.out.println("ERROR IN DEMO");
            Logger.getLogger(ACF.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
