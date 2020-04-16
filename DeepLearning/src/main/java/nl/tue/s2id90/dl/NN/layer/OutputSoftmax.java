package nl.tue.s2id90.dl.NN.layer;

import nl.tue.s2id90.dl.NN.activation.Softmax;
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
public class OutputSoftmax extends FullyConnected implements OutputLayer{
    
    // loss function for calculating back-propagation
    private final Loss              loss;
    
    /**
     * 
     * @param name
     * @param input_shape 
     * @param classes 
     * @param loss 
     */
    public OutputSoftmax( String name, TensorShape input_shape, int classes, Loss loss ){        
        super(name, input_shape, classes, new Softmax() );
        
        this.loss = loss;        
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

    /**
     * Calculate loss over label and prediction
     * 
     * @param label      tensor with correct output
     * @return loss value
     */
    @Override
    public double calculateLoss( Tensor label ) {
        
        return loss.calculate_loss( label.getValues(), last_preoutput, activation);
    }  
    
    @Override
    public JSONObject json() {
        JSONObject jo = super.json();
        jo.put("loss", loss.json());
        return jo;
    }
}
