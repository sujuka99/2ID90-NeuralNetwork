package nl.tue.s2id90.dl.NN.layer;

import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Roel van Engelen
 */
public class InputLayer extends Layer{
        
    private final boolean show_values;
    
    /**
     * 
     * @param name
     * @param shape_input 
     */
    public InputLayer( String name, TensorShape shape_input ){
        super(name, shape_input, shape_input );
        
        show_values = false;
    }
    
    /**
     * 
     * @param name
     * @param shape_input 
     * @param show_values 
     */
    public InputLayer( String name, TensorShape shape_input, boolean show_values ){
        super(name, shape_input, shape_input );
        
        this.show_values = show_values;
    }
        
    /**
     * should activation image show actual values or relative activations
     * 
     * @return true for actual values, false for relative activations
     */
    @Override
    public boolean showValues(){
        
        return show_values;
    }
    
    /**
     * Initialize bias and weights
     * 
     * @param initializer 
     */
    @Override
    public void initializeLayer( Initializer initializer ){
        
        // Nothing to initialize
    }
    
    /**
     * Calculate inference of input tensor
     * @param input
     * @return 
     * @throws nl.tue.s2id90.dl.NN.error.IllegalInput 
     */
    @Override
    public Tensor inference( Tensor input ) throws IllegalInput{
        
        // verify correct Tensor input shape        
        if( !input.isCorrectShape(inputShape ) ){ 
            String message = String.format(
                    "Layer \"%s\" expected shape %s, but got tensor with shape %s and array with shape %s",
                    getName(),
                    inputShape.shapeToString(),
                    input.getShape().shapeToString(),
                    input.getValues().shapeInfoToString()
            );
            throw new IllegalInput(message );
        } else {
            // Huub: debug statements (should be commented out, eventually)
//            String message = String.format(
//                    "Layer \"%s\" expected shape %s, and got tensor with shape %s and array with shape %s",
//                    get_name(),
//                    shape_input.shape_to_string(),
//                    input.get_shape().shape_to_string(),
//                    input.get_values().shapeInfoToString()
//            );
//            System.out.println(message);
            
            //System.out.println( input.get_values().shapeInfoToString() );
            //throw new Illegal_Input( shape_input.shape_to_string() + " " + input.get_shape().shape_to_string() );
        }
        
        return input;
    }

    /**
     * Calculate back-propagation
     * @param input
     * @return 
     */
    @Override
    public INDArray backpropagation( INDArray input ) {
        
        // input layer, no changes will be propagated
        
        return input;
    }
    
    @Override
    public JSONObject json() {
        JSONObject jo = super.json();
        jo.put("show_values", show_values);
        return jo;
    }
}