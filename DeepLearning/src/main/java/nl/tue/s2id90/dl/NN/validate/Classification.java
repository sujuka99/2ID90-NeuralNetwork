package nl.tue.s2id90.dl.NN.validate;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Classification
 * classification validation: the argmax of two tensor should
 * be equal for the classification to be correct.
 * 
 * @author Roel van Engelen
 */
public class Classification implements Validator{
    
    /**
     * Calculate how accurate the prediction is
     * classification so it is either good or wrong, 1.0 or 0.0
     * 
     * @param label      tensor with correct classification
     * @param prediction tensor with predicted classification
     * @return           % how accurate prediction is 1.0 or 0.0
     */
    @Override
    public Float validate( Tensor label, Tensor prediction ){
        return accuracy(label.getValues(),prediction.getValues());
    }
    
    private Float accuracy( INDArray label, INDArray prediction ){
                        
        float correct = 0;
        for( int x = 0 ; x < label.shape()[0] ; x++ ){
                                    
            if( arg_max( label.getRow( x ) ) == arg_max( prediction.getRow( x ) ) ){
                
                correct += 1;
            }
        }
                
        // incorrect prediction
        return correct / label.shape()[0];
    }
    
    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private
        
    /**
     * find classification with the highest value 
     * 
     * @param data INDarray to search
     * @return     location of max value
     */
    private int arg_max( INDArray data ){
        int   ret = 0;
        float value = -Float.MAX_VALUE;
        
        for( int x = 0 ; x < data.length() ; x++ ){
            
            if( data.getFloat( x ) > value ){
                
                ret = x;
                value = data.getFloat( x );
            }
        }
        
        return ret;
    }
}
