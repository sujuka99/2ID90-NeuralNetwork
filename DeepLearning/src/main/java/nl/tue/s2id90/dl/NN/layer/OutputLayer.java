package nl.tue.s2id90.dl.NN.layer;

import nl.tue.s2id90.dl.NN.tensor.Tensor;

/**
 *
 * @author Roel van Engelen
 */
public interface OutputLayer{
    
    /**
     * Calculate loss over label and prediction
     * 
     * @param labels      tensor with correct output
     * @return loss value
     */
    public double calculateLoss( Tensor labels );
}
