package nl.tue.s2id90.dl.NN.loss;

import nl.tue.s2id90.dl.NN.activation.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 *
 * @author Roel van Engelen
 */
public class CrossEntropy implements Loss{
    
    // Nd4j loss function
    private final ILossFunction loss;
    
    public CrossEntropy(){
        
        // initialize Nd4j loss function
        loss = LossFunction.MCXENT.getILossFunction();
    }
    
    /**
     * Calculate loss over label and prediction
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @param activation  ND4J Iactivation type
     * @return            cross entropy loss value
     */
    @Override
    public double calculate_loss( INDArray labels, INDArray preoutput, Activation activation ){
        
        return loss.computeScore( labels, preoutput, activation.get_IActivation(), null, true );
    }
    
    /**
     * calculate final layer cross entropy backpropagation gradient
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
