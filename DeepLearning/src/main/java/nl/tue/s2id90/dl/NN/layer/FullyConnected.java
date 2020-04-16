package nl.tue.s2id90.dl.NN.layer;

import static java.lang.String.format;
import java.util.Map;
import nl.tue.s2id90.dl.NN.activation.Activation;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import java.util.function.Supplier;
import nl.tue.s2id90.dl.Nd4jUtil;
import nl.tue.s2id90.dl.json.JSONUtil;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class FullyConnected extends Layer{
        
    protected     INDArray weights;
    protected     INDArray weights_flatten;
    protected     INDArray bias;    
    protected     INDArray updated_weights;
    protected     INDArray updated_bias;
    
    protected     INDArray last_input;
    protected     INDArray last_preoutput;
    
    private final int inputs;
    private final int outputs;
    protected final Activation activation;
    
    /**
     * Create Fully connected layer
     * 
     * @param name        layer name
     * @param activation  activation function
     * @param input_shape 1D Tensor input shape 
     * @param outputs     amount of outputs
     */
    public FullyConnected( String name, TensorShape input_shape, int outputs, Activation activation){
        super(name, input_shape, new TensorShape( outputs ) );
        
        // check that input shape is 1D
        if( input_shape.is3D() ){
            throw new IllegalArgumentException(format("Layer \"%s\" only accepts 1D input shapes, but is constructed with input shape: %s.", name, input_shape));
        }
        
        this.outputs    = outputs;
        this.activation = activation;
        this.inputs     = input_shape.getNeuronCount();
    }
        
    /**
     * should activation image show actual values or relative activations
     * 
     * @return true for actual values, false for relative activations
     */
    @Override
    public boolean showValues(){
        
        return false;
    }
    
    /**
     * Initialize bias and weights
     * 
     * @param initializer 
     */
    @Override
    public void initializeLayer( Initializer initializer ){
                
        // initialize bias and weights INDArrays
        // init bias with value zero
        bias    = Nd4j.zeros( new int[] { outputs } );
        // init weights with initializer, keep a flattend weights and view with correct shape
        // initialize with 'f' ordering because of ND4j.gemm in backpropagation
        weights_flatten = initializer.get_weight(inputs, outputs, new int[] { inputs, outputs } );
        weights = weights_flatten.reshape( 'f', new int[] { inputs, outputs } );
        
        // initialize bias and weight gradient accummulation INDArrays
        updated_bias    = Nd4j.zeros(new int[] { outputs}); //bias.dup();                    // Huub: explicitly set to zero
        updated_weights = Nd4j.zeros(new int[] { inputs, outputs},'f'); //weights.dup();     // Huub: explicitly set to zero
    }
    
    /**
     * Calculate inference on input tensor
     * 
     * @param input
     * @return 
     */
    @Override
    public Tensor inference( Tensor input ){                                   // forward pass
        if (weights==null) {
            throw new IllegalStateException(
                format("The weights of layer \"%s\" have not been initialized!", getName())
            );
        }
                                      
        // duplicate input, backpropagation needs it
        if (isInTrainingMode())
            last_input = input.getValues().dup();
        
        // calculate inference
        INDArray ret = input.getValues().mmul(weights );                      // ret <-- input*weights  (apparently input and ret are  row vectors)
        
        // add bias
        ret = ret.addiRowVector( bias );                                        // ret <-- input*weights + bias  
        
        // duplicate pre activation output, backpropagation needs it
        if (isInTrainingMode())
            last_preoutput = ret.dup();
        
        // calculate activation
        activation.activation( ret );                                            // ret <-- activation(ret)    // apply activation function to each element of ret
                               
        // return inference tensor
        return new Tensor( ret, new TensorShape( outputs ) );
    }

    /**
     * Calculate back-propagation
     * 
     * @param epsilon
     * @return 
     */
    @Override
    public INDArray backpropagation( INDArray epsilon ) {
        
        // backpropagation is copied from dl4j:
        // https://github.com/deeplearning4j/deeplearning4j/blob/cc5ea0dddd5ed61af26cee1ec56d144d1b70b450/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/layers/BaseLayer.java
        // function: backpropGradient - line: 70
        
        // calculate delta
        INDArray delta = activation.get_IActivation().backprop( last_preoutput, epsilon ).getFirst();
                                                                                // Huub:
                                                                                // Let f be the activation function, a = f(z)
                                                                                // epsilon = dL/da, last_preoutput=z
                                                                                // delta <-- dL/dz = dL/da * da/dz = dL/da * grad(f wrt z)
        
        // calculate epsilon for next layer (error)                
        INDArray epsilon_next = weights.mmul( delta.dup().transpose() ).transpose();
                                                                                // Huub: epsilon_next <-- (weights * delta^T)^T 
        
        // calculate weight update
        // updated_weights has to be in 'f' ordering 
        Nd4j.gemm( last_input, delta, updated_weights, true, false, 1.0, 0.0 ); // Huub: updated_weights <-- 1.0 * last_input^T * delta  +  0.0 * updated_weights ?????????
        
        // calculate bias update
        delta.sum( updated_bias, 0 );
        
        return epsilon_next;
    }
//
//    /**
//     * Update bias and weights
//     * 
//     * @param learning_rate
//     * @param batch_size
//     */
//    @Override
//    public void updateLayer( float learning_rate, int batch_size ){
//                
//        // update bias with accumulated gradients
//        updated_bias.muli( learning_rate / batch_size );
//        
//        // dl4j uses a stepfunction to update sgd weights and bias
//        // https://github.com/deeplearning4j/deeplearning4j/blob/1f8af820c29cc5567a2c5eaa290f094c4d1492a7/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/stepfunctions/NegativeDefaultStepFunction.java
//        // where step direction is -1 with sgd
//        // TODO add direction parameter to update_layer so that different optimizers can be used
//        
//        // update bias
//        Nd4j.getBlasWrapper().level1().axpy( bias.length(), -1.0, updated_bias, bias );
//        // re initialize update_bias array
//        updated_bias = Nd4j.zeros( new int[] { outputs } );
//        
//        // update weights with accumulated gradients
//        updated_weights.muli( learning_rate / batch_size );
//        
//        //weights.subi( updated_weights );
//                
//        // dl4j uses a stepfunction to update sgd weights and bias
//        // https://github.com/deeplearning4j/deeplearning4j/blob/1f8af820c29cc5567a2c5eaa290f094c4d1492a7/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/stepfunctions/NegativeDefaultStepFunction.java
//        // where step direction is -1 with sgd
//        
//        // update wights
//        Nd4j.getBlasWrapper().level1().axpy( weights_flatten.length(), -1.0, updated_weights, weights_flatten );
//        // re initialize update_weights array
//        updated_weights.assign(0); //weights.dup();                             // Huub: explicitly set to zero
//        //updated_weights = weights.dup();
//    }

    UpdateFunction biasUpdater, weightsUpdater;
    @Override
    public void updateLayer(Supplier<UpdateFunction> createUpdateFunction, double learning_rate, int batch_size) {
        if (biasUpdater==null) biasUpdater=createUpdateFunction.get();
        if (weightsUpdater==null) weightsUpdater=createUpdateFunction.get();
        
        // duplicating the gradients, since update() method might change them otherwise.
        INDArray biasGrad = updated_bias.dup(updated_bias.ordering());
        biasUpdater   .update(bias   , true , learning_rate, batch_size, biasGrad);
//        updated_bias.assign(0);
        
        INDArray weightsGrad = updated_weights.dup(updated_weights.ordering());
        weightsUpdater.update(weights, false, learning_rate, batch_size, weightsGrad);
//        updated_weights.assign(0);
    }
    
//<editor-fold defaultstate="collapsed" desc="serialization">
    
    @Override
    public JSONObject json() {
        JSONObject jo = super.json();
        jo.put("activation", activation.json());
        jo.put("weights",JSONUtil.toJson(weights_flatten));
        jo.put("bias",JSONUtil.toJson(bias));
        return jo;
    }
    /** non-public, package method to set weights directly, used during
     * de-serialization.
     */
    void setWeights(float[] w) {
        int expectedSize =inputs*outputs;
        if (w.length!= expectedSize) {
            throw new IllegalArgumentException(format("Weights array has length %d, but expected %d!", w.length, expectedSize));
        }
        //        weights_flatten = new NDArray(w);
        weights_flatten = Nd4j.create(w);
        weights = weights_flatten.reshape( 'f', new int[] { inputs, outputs } );
    }
    
    /** @return weights as a flat nd array. Returns a copy. */
    public INDArray getWeightsAsNDArray() {
        return weights_flatten.dup();
    }
    
    /** non-public, package method to set biases directly, used during
     * de-serialization.
     */
    void setBias(float[] w) {
        if (w.length!= outputs) {
            throw new IllegalArgumentException(format("Bias array has length %d, but expected %d!",w.length,outputs));
        }
        //        bias = new NDArray(w);
        bias = Nd4j.create(w);
    }
//</editor-fold>

    @Override
    public Map<String, Object> getInfoMap() {
        Map result= super.getInfoMap();
        result.put("activation", activation.getClass().getSimpleName());
        return result;
    }
    
    //<editor-fold defaultstate="collapsed" desc="auxiliary functions for GradientChecker">
    @Override
    INDArray getGradient() {
        INDArray flatWeights = updated_weights.reshape( 'f', new int[] { 1, inputs*outputs } );
        return Nd4j.concat(1, flatWeights, updated_bias);
    }

    @Override
     Double getParameter(int i) {
        return Layers.getParameter(i, weights_flatten, bias);
    }
    
    @Override
     boolean setParameter(int i, double d) {
        return Layers.setParameter(i, d, weights_flatten, bias);
    }
    
    @Override
     int getNumberOfParameters() {
        return Layers.getNumberOfParameters(weights_flatten, bias);
    }
     
    @Override
     void clearParameters() {
        Layers.clearParameters(weights_flatten, bias);
    }
    //</editor-fold>
}
