package nl.tue.s2id90.dl.NN.layer;

import java.util.LinkedHashMap;
import java.util.Map;
import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import java.util.function.Supplier;
import lombok.Getter;
import lombok.Setter;
import nl.tue.s2id90.dl.json.JSONable;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * ...
 * 
 * Some layers, e.g. dropout layer, behave differently during training than 
 * during inference. This is regulated via set/getTrainingMode boolean.
 * @author Roel van Engelen
 */
public abstract class Layer implements JSONable {
    
    // layer name
    @Getter private   final String       name;
    @Getter protected final TensorShape inputShape;
    @Getter protected final TensorShape outputShape; 
    @Getter @Setter protected boolean inTrainingMode=false;
    
    /**
     * 
     * @param name
     * @param shape_input
     * @param shape_output 
     */
    public Layer(String name, TensorShape shape_input, TensorShape shape_output ){
        this.name         = name;
        this.inputShape  = shape_input;
        this.outputShape = shape_output;
    }
    
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////// abstract
    
    /**
     * should activation image show actual values or relative activations
     * 
     * @return true for actual values, false for relative activations
     */
    public abstract boolean showValues();
    
    /**
     * Initialize bias and weights
     * 
     * @param initializer 
     */
    public abstract void initializeLayer( Initializer initializer );
    
    /**
     * Calculate inference of input tensor
     * 
     * @param input
     * @return 
     * @throws nl.tue.s2id90.dl.NN.error.IllegalInput 
     */
    public abstract Tensor inference( Tensor input ) throws IllegalInput;
    
    /**
     * Calculate back-propagation
     * 
     * @param input
     * @return 
     */
    public abstract INDArray backpropagation( INDArray input );
    
//    /**
//     * Update bias and weights
//     * 
//     * @param learning_rate
//     * @param batch_size
//     * @deprecated 
//     */
//    public abstract void updateLayer( float learning_rate, int batch_size );
    
     /**
     * Update bias and weights
     * 
     * @param createUpdateFunction
     * @param learning_rate
     * @param batch_size
     */
    public void updateLayer(Supplier<UpdateFunction> createUpdateFunction,  double learning_rate, int batch_size  ) {
        // By default do nothing, only layers with updatable weights/biases should implement this function.
    }

    @Override
    public JSONObject json() {
        JSONObject jo = JSONable.super.json();
        jo.put("name", name);
        jo.put("input_shape", inputShape.json());
        jo.put("output_shape", outputShape.json());
        return jo;
    }  
    
     /** @return a map with named informational objects for this model. */
    public Map<String,Object> getInfoMap() {
        Map result = new LinkedHashMap<>();
        result.put("name", getName());
        result.put("input shape",inputShape.toString());
        result.put("output shape", outputShape.toString());
        return result;
    }
    
    //<editor-fold defaultstate="collapsed" desc="auxiliary functions for GradientChecker">
    INDArray getGradient() {
        return null; // by default no parameters for updating by gradient descent
    }
    
    /** @return returns the value of parameter i; if the layer has less
     *          than i parameters, null is returned.
     */
    Double getParameter(int i) {
        return null;  // by default this parameter is not there
    }

    /** sets parameter i to value d.
     * @param i   index of parameter
     * @param d   value to be set
     * @return true, if parameter i exists, false otherwise.
     */    
    boolean setParameter(int i, double d) {
        return false;
    }
    
    int getNumberOfParameters() {
        return 0;    // by default no parameters
    }
    
    void clearParameters() {
        // by default no Parameters, so nothing to do.
    }
    //</editor-fold>
    
    @Override
    public String toString() {
        return getInfoMap().toString();
    }
}
