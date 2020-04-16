package nl.tue.s2id90.dl.NN.layer;

import java.util.Map;
import nl.tue.s2id90.dl.NN.activation.Identity;
import nl.tue.s2id90.dl.NN.loss.Loss;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class SimpleOutput extends FullyConnected implements OutputLayer{

    private final boolean show_values;
    
    // loss function for calculating back-propagation
    private final Loss loss;
    
    /**
     * 
     * @param name
     * @param input_shape
     * @param outputs
     * @param loss 
     */
    public SimpleOutput( String name, TensorShape input_shape, int outputs, Loss loss ){
        super(name, input_shape, outputs, new Identity() );
        
        this.loss   = loss;
        show_values = false;
    }
    
    /**
     * 
     * @param name
     * @param input_shape
     * @param outputs
     * @param loss
     * @param show_values 
     */
    public SimpleOutput( String name, TensorShape input_shape, int outputs, Loss loss, boolean show_values ) {
        super(name, input_shape, outputs, new Identity() );
        
        this.loss        = loss;
        this.show_values = show_values;
    }

    /**
     * Calculate loss over label and prediction
     * 
     * @param labels      tensor with correct output
     * @return loss value
     */
    @Override
    public double calculateLoss( Tensor labels ){
        
        return loss.calculate_loss( labels.getValues(), last_preoutput, activation);
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
     * Calculate back-propagation
     * 
     * @param correct_labels INDArray with correct labels
     * @return epsilon
     */
    @Override
    public INDArray backpropagation( INDArray correct_labels ) {
            
        // this is a final layer, calculate delta with loss function
        // see: https://github.com/deeplearning4j/deeplearning4j/blob/1f8af820c29cc5567a2c5eaa290f094c4d1492a7/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/BaseOutputLayer.java
        // function: getGradientsAndDelta - line: 174
        INDArray delta = loss.computeGradient( correct_labels, last_preoutput, activation);
        
        // see Fully_Connected for full explanation
        // function: backpropagation - line: 120
        
        // calculate weight update
        Nd4j.gemm( last_input, delta, updated_weights, true, false, 1.0, 0.0 );
        
        // calculate bias update
        delta.sum( updated_bias, 0 );
        
        // calculate epsilon for next layer (error)
        INDArray epsilon_next = weights.mmul( delta.dup().transpose() ).transpose();
        
        return epsilon_next;
    }
    
    
    @Override
    public JSONObject json() {
        JSONObject jo = super.json();
        jo.put("loss", loss.json());
        jo.put("show_values",show_values);
        return jo;
    }
    
    @Override
    public Map<String, Object> getInfoMap() {
        Map result= super.getInfoMap();
        result.put("loss", loss.getClass().getSimpleName());
        return result;
    }
}
