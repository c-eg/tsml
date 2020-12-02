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

package tsml.data_containers.utilities;

import java.util.ArrayList;
import java.util.List;

import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;


//This class if for weird hacky dimension wise operations we need to do when interfacing with Weka classifiers
//that can only take univariate data.
public class Splitter{

    
    /** 
     * @param inst
     * @return List<TimeSeriesInstance>
     */
    //splitty splitty.
    public static List<TimeSeriesInstance> splitTimeSeriesInstance(TimeSeriesInstance inst){
        List<TimeSeriesInstance> output = new ArrayList<>(inst.getNumDimensions());

        for(TimeSeries ts : inst){
            double[][] wrapped_raw = new double[1][];
            wrapped_raw[0] = ts.toArray();

            output.add(new TimeSeriesInstance(wrapped_raw, inst.getLabelIndex()));
        }

        return output;
    }

    
    /** 
     * @param inst
     * @return List<TimeSeriesInstances>
     */
    //horizontally slice into univariate TimeSeriesInstances.
    public static List<TimeSeriesInstances> splitTimeSeriesInstances(TimeSeriesInstances inst){
        List<TimeSeriesInstances> output = new ArrayList<>(inst.getMaxNumChannels());

        for(int i=0; i< inst.getMaxNumChannels(); i++){
            TimeSeriesInstances temp = new TimeSeriesInstances(inst.getHSliceArray(new int[]{i}), inst.getClassIndexes());
            temp.setClassLabels(inst.getClassLabels());
            output.add(temp);
        }

        return output;
    }

    
    /** 
     * @param inst_dims
     * @return TimeSeriesInstance
     */
    //mergey mergey
    public static TimeSeriesInstance mergeTimeSeriesInstance(List<TimeSeriesInstance> inst_dims){
        double[][] wrapped_raw = new double[inst_dims.size()][];
        int i=0;
        for(TimeSeriesInstance inst : inst_dims){
            //ignore any other dimensions, because they should only be single 
            wrapped_raw[i++] = inst.toValueArray()[0]; 
        }   

        return new TimeSeriesInstance(wrapped_raw, inst_dims.get(0).getLabelIndex());
    }
    
}