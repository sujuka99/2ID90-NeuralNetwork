package nl.tue.s2id90.dl.NN.activation;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class Sigmoid implements Activation{

    private final INDArray          epsilon;
    private final ActivationSigmoid sigmoid;
   
    /**
     * creates a sigmoid activation function: <pre>
     *          1
     *      --------
     *            -x
     *       1 + e
     * </pre>.
     */
    public Sigmoid(){
        
        epsilon = Nd4j.create( new float[] { 1.0f }, new int[] { 1 }, 'c' );
        sigmoid = new ActivationSigmoid();
    }
    
        
    /**
     * Apply activation to INDArray tensor
     * Sigmoid activation function defined as
     *        1
     *      --------
     *            -x
     *       1 + e
     * 
     * @param tensor INDArray activation has to be applied to
     */
    @Override
    public void activation( INDArray tensor ){
        
        sigmoid.getActivation(tensor, true);
    }
    
    
    /**
     * Calculate derivative of sigmoid function defined as
     * x * ( 1.0 - x )
     * 
     * @param tensor INDArray derivate has to be calculated of
     */
    @Override
    public INDArray derivative( INDArray tensor ){
        
        return sigmoid.backprop( tensor, epsilon ).getKey();
    }
    
    /**
     * Calculate activation backprop
     * 
     * @param preoutput INDArray preoutput
     * @param epsilon
     */
    @Override
    public INDArray backpropagation( INDArray preoutput, INDArray epsilon ){
                
        return sigmoid.backprop( preoutput, epsilon ).getKey();
    }
    
    /**
     * 
     * @return 
     */
    @Override
    public IActivation get_IActivation(){
        
        return sigmoid;
    }
}
