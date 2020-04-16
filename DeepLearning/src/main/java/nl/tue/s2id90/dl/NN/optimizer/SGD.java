package nl.tue.s2id90.dl.NN.optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import nl.tue.s2id90.dl.NN.Model;
import java.util.function.Supplier;
import lombok.Builder;
import lombok.NonNull;
import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.optimizer.update.GradientDescent;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.experiment.BatchResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import nl.tue.s2id90.dl.NN.validate.Validator;
import nl.tue.s2id90.dl.Util;

/**
 * SGD
 * 
 * @author Roel van Engelen
 */
public class SGD extends Optimizer{
    @NonNull private Supplier<UpdateFunction> updateFunctionConstructor=GradientDescent::new;
     
    /**
     * Initialize new SGD optimizer
     * 
     * @param model         Model to be trained
     * @param learningRate learning rate
     * @param validator     model validator
     * @param updateFunction a constructor for an update function, e.g. GradientDescent::new 
     */
    @Builder private SGD(@NonNull Model model, Double learningRate, Validator validator, Supplier<UpdateFunction> updateFunction ){
        super();
        
        this.model                       = model;
        if (validator!=null) this.validator = validator;
        if (learningRate!=null) this.learningRate = learningRate; else this.learningRate = 1;  // assign a non-zero value to learningRate!
        if (updateFunction!=null) this.updateFunctionConstructor = updateFunction;
    }
    
    /**
     * Train model with single batch
     * 
     * @param batch List with Tensor_Pair training batch
     * @return 
     * @throws IllegalInput
     */
    @Override
    public BatchResult trainOnBatch( TensorPair batch ) throws IllegalInput{
        
        int batch_size = (int)batch.model_input.getValues().shape()[0];        
        
        if (!model.isInTrainingMode()) {
            throw new IllegalStateException("Model trained while not in training Mode");
        }
        
        // calculate inference
        Tensor prediction = model.inference( batch.model_input );

        // check if prediction is correct
        Float accuracy = validator.validate( batch.model_output, prediction );

        // calculate loss
        double loss = model.getOutputLayer().calculateLoss(batch.model_output );

        // list with all layers
        List<Layer> layers = model.getLayers();

        // calculate gradients, using backpropagation starting from output layer.
        INDArray back = batch.model_output.getValues();

        // loop over all layers from ( last -1 ) to first layer
        for( int x = layers.size() - 1 ; x > 0 ; x-- ){

            back = layers.get( x ).backpropagation( back );
        }
        
        // update layer weights with calculated gradients
        for( Layer layer : model.getLayers() ){
            
            layer.updateLayer(updateFunctionConstructor, learningRate, batch_size );
        }
                
        return new BatchResult( ++batchId, accuracy, loss, learningRate );
    }
    

    
    /** @return a map with named informational objects for this model. */
    @Override
    public Map<String,Object> getInfoMap() {
        Map result = new LinkedHashMap<>();
        result.put("optimizer", "Stochastic Gradient Descent");
        result.put("Validator", validator.getClass().getSimpleName());
        result.put("update function", updateFunctionConstructor.get().info());
        result.put("learning rate", learningRate);
        return result;
    }
    
    @Override
    public String toString() {
        return Util.toString(getInfoMap());
    }
}
