package nl.tue.s2id90.dl.NN.validate;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Regression
 * Regression validation implementation using sqrt of mean squared error
 * 
 * @author Roel van Engelen
 */
public class Regression implements Validator{
    
    /**
     * Calculate how accurate the prediction is
     * for regression we use root mean squared error. So, here lower is better.
     * 
     * @param label      tensor with correct values
     * @param prediction tensor with predicted values
     * @return           mse how accurate prediction is
     */
    @Override
    public Float validate( Tensor label, Tensor prediction ){
        // ToDo: rename this method: accuracy suggests that higher is better, but in this case that is not so!
        // duplicate label INDArray
        INDArray se = label.getValues().dup();     
        // substract prediction = error
        se.subi( prediction.getValues() );
        // raise each value to the power 2
        se = Transforms.pow( se, 2 );    // ToDo: replace by muli ??
        
        // calculate mean, returning the mean squared error
        return Nd4j.mean( se ).getFloat( 0 );
    }
}
