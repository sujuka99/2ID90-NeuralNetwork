package nl.tue.s2id90.dl.NN.optimizer;

import nl.tue.s2id90.dl.experiment.BatchResult;
import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.Setter;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.validate.Validator;
import nl.tue.s2id90.dl.NN.validate.ZeroValidator;

/**
 * Optimizer
 * Abstract Optimizer class this ensures new optimizer implementation will
 * have all functions required to train a model
 * 
 * also contains all Gui functions, passing data to gui to be shown
 * 
 * @author Roel van Engelen
 */
public abstract class Optimizer {
    
    @Getter @Setter protected double learningRate;
    @Getter protected  Model    model = null;
    @Getter protected  Validator validator = new ZeroValidator();
    @Getter protected int batchId;
    
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////// abstract
    
    /**
     * Train model with single batch
     * 
     * @param batch List with Tensor_Pair training batch
     * @return 
     * @throws IllegalInput
     */
    public abstract BatchResult trainOnBatch( TensorPair batch ) throws IllegalInput;
    
    /**
     * validate model accuracy
     * 
     * @param batch List with Tensor_Pair validation batch
     * @return 
     * @throws IllegalInput
     */
    public BatchResult validate(List<TensorPair> batch ) throws IllegalInput{
        
        if (model.isInTrainingMode()) {
            throw new IllegalStateException("Model validated while in training Mode");
        }
        
        float accuracy = 0;
        
        // loop over batch and calculate accuracy
        for( TensorPair sample : batch ){
            
            // predict image classification
            Tensor output = model.inference( sample.model_input );
            
            // check if prediction is correct
            accuracy += validator.validate( sample.model_output, output );
        }
        
        // calculate accuracy
        // accuracy = correct predictions / batch size
        accuracy /= batch.size();
        
        // update validation graph
        // loss and learning rate are irrelevant
        return new BatchResult( batchId, accuracy);
    }
    
    abstract public Map<String,Object> getInfoMap();
}
