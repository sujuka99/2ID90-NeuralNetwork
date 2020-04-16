package nl.tue.s2id90.dl.NN.validate;

import nl.tue.s2id90.dl.NN.tensor.Tensor;

/**
 * Validate
 * Abstract validation interface
 * different types of models/data require a different type of validation
 * 
 * @author Roel van Engelen
 */
public interface Validator {
    
    /**
     * Calculate a value for the prediction given the true label.
     * 
     * @param label      tensor with correct output
     * @param prediction tensor with predicted output
     * @return           value indication how accurate prediction is
     */
    public Float validate( Tensor label, Tensor prediction );    
}
