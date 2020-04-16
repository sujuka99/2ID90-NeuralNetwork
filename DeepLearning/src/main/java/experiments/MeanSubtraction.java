package experiments;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.transform.DataTransform;

public class MeanSubtraction implements DataTransform{
    Double mean = 0.0;//mean of all images
    
    @Override
    public void fit (List<TensorPair> data) {
        Double sum = 0.0;//sum of the values from all images
        Double numOfElements = 0.0;//number of values
        
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Empty dataset");
        }
        for (TensorPair pair: data){
            //first we get input from the model and save it
            Tensor save = pair.model_input;
            
            //add all the values from current TensorPair, needs to be of type double
            sum = sum + save.getValues().sumNumber().doubleValue();
            
            //add the number of elements of the current TensorPair
            numOfElements = numOfElements + save.getValues().length();
        }
        //calculate mean 
        mean = sum/numOfElements;
    }
    
    @Override
    public void transform (List<TensorPair> data) {
        for (TensorPair pair: data){
            //first we get input from the model and save it
            Tensor save = pair.model_input;
            
            //then we subtract the mean of colors from the current data pair
            //we use subi method, hence we change the current save rather than create a new one
            save.getValues().subi(mean);
        }
    }
}
