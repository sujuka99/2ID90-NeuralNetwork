package nl.tue.s2id90.dl.NN.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class Softmax implements Activation{

    private final INDArray       epsilon;
    private final ActivationSoftmax softmax;
    
    /**
     * 
     * f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift), where shift = max_i(x_i)
     */
    public Softmax(){
        
        epsilon = Nd4j.create( new float[] { 1.0f }, new int[] { 1 }, 'c' );
        softmax = new ActivationSoftmax(); 
    }
    
    
    /**
     * Apply activation to INDArray tensor
     * RELU activation function defined as
     * max( 0, activation )
     * 
     * @param tensor INDArray activation has to be applied to
     */
    @Override
    public void activation( INDArray tensor ){
                
        softmax.getActivation( tensor, true );
    }
    
    
    /**
     * Calculate derivative and apply to tensor
     * 
     * @param tensor INDArray derivate has to be calculated of
     */
    @Override
    public INDArray derivative( INDArray tensor ){
                
        return softmax.backprop( tensor, epsilon ).getKey();
    }
    
    /**
     * Calculate activation backprop
     * 
     * @param preoutput INDArray preoutput
     * @param epsilon
     */
    @Override
    public INDArray backpropagation( INDArray preoutput, INDArray epsilon ){
                
        return softmax.backprop( preoutput, epsilon ).getKey();
    }
    
    /**
     * 
     * @return 
     */
    @Override
    public IActivation get_IActivation(){
        
        return softmax;
    }
}