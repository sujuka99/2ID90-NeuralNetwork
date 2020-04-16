package nl.tue.s2id90.dl.NN.loss;

import nl.tue.s2id90.dl.NN.activation.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *
 * @author Roel van Engelen
 */
public class L1 implements Loss{
    
    // Nd4j loss function
    private final ILossFunction loss;
    
    public L1(){
        
        // initialize Nd4j loss function
        loss = LossFunctions.LossFunction.L1.getILossFunction();
    }
    
    /**
     * Calculate loss over label and prediction
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @return            mse loss value
     */
    @Override
    public double calculate_loss( INDArray labels, INDArray preoutput, Activation activation ){
        
        return loss.computeScore( labels, preoutput, activation.get_IActivation(), null, true );
    }
    
    /**
     * calculate final layer MSE backpropagation gradient
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @param activation  ND4J Iactivation type
     * @return            INDArray backpropagation gradient
     */
    @Override
    public INDArray computeGradient( INDArray labels, INDArray preoutput, Activation activation ){
        
        return loss.computeGradient( labels , preoutput, activation.get_IActivation(), null );
    }
}
