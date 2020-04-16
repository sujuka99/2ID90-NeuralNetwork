package nl.tue.s2id90.dl.NN.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class LRELU implements Activation{

    private final INDArray       epsilon;
    private final ActivationLReLU lrelu;
    
    /**
     * creates  a 'rectified linear unit' activation function: relu(x)=ax(0,x).
     */
    public LRELU(){
        
        epsilon = Nd4j.create( new float[] { 1.0f }, new int[] { 1 }, 'c' );
        lrelu = new ActivationLReLU(); 
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
                
        lrelu.getActivation( tensor, true );
    }
    
    
    /**
     * Calculate derivative and apply to tensor
     * 
     * @param tensor INDArray derivate has to be calculated of
     */
    @Override
    public INDArray derivative( INDArray tensor ){
                
        return lrelu.backprop( tensor, epsilon ).getKey();
    }
    
    /**
     * Calculate activation backprop
     * 
     * @param preoutput INDArray preoutput
     * @param epsilon
     */
    @Override
    public INDArray backpropagation( INDArray preoutput, INDArray epsilon ){
                
        return lrelu.backprop( preoutput, epsilon ).getFirst();
    }
    
    /**
     * 
     * @return 
     */
    @Override
    public IActivation get_IActivation(){
        
        return lrelu;
    }
}