package nl.tue.s2id90.dl.experiment;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import java.util.List;

/**
 * Activation_Data
 * Convenience class holding all activation tensors, names and settings per layer
 * 
 * @author Roel van Engelen
 */
public class ActivationData{
    
    public final int                 batch_id;
    
    public final List<String>        names;
    public final List<Tensor>        tensors;
    public final List<Boolean>       show_values;
    
    /**
     * Convenience class holding all activation tensors, names and settings per layer
     * 
     * @param batch_id    current batch id
     * @param names       List with layer names
     * @param tensors     List with layer feedforward activation tensors
     * @param show_values List with show_values setting per layer
     *                    if true  the image shows the actual activation value
     *                    if false the image shows relative activations
     */
    public ActivationData( int batch_id, List<String> names, List<Tensor> tensors, List<Boolean> show_values ){
        
        this.batch_id    = batch_id;
        this.names       = names;
        this.tensors     = tensors;
        this.show_values = show_values;
    }   
    
    @Override
    public String toString() {        // used as label in FXML Activation menu item
        return "Activation at batch " + batch_id;
    }
}
