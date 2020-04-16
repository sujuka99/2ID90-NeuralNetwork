package nl.tue.s2id90.dl.NN.initializer;

import nl.tue.s2id90.dl.NN.layer.Layer;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Roel van Engelen
 */
public interface Initializer {
        
    /**
     * Initialize all layers in layers
     * 
     * @param layers list with layers
     */
    default public void initialize_layers( List<Layer> layers ) {
        
        // loop over all layers and initialize all weights and biases
        layers.forEach( ( layer ) -> {
            
            layer.initializeLayer( this );
        });
    }
    
    /**
     * create flat weight array for the given shape. 
     * @param fanIn
     * @param fanOut
     * @param shape
     * @return 
     */
    public INDArray get_weight(int fanIn, int fanOut, int[] shape );
    
}