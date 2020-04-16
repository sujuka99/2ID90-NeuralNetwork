/*
 * Gradient descent with momentum is an approach that often yields better
 * better convergence on deep networks
 */
package experiments;

import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Implementation of Gradient Descent with Momentum.
 * @author Lyuben Petrov
 * @author Ivan Yordanov
 */
public class GradientDescentMomentum implements UpdateFunction {
    INDArray update;
    float mu = 0.5f;//example value
    //float[] mu_vals = new float[]{0.5f, 0.9f, 0.95f, 0.99f};//usual values for the momentum
    
    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
     * @param value
     * @param gradient
     * @param learningRate
     * @param batchSize
     * @param isBias
     */
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        //on the first call of this method, create update vector
        if (update == null) update = gradient.dup('f').assign(0);
        
        /*
        * Implements momentum as given in CS231n part 3:
        *   v = mu * v - learning_rate * dx : velocity
        *   x += v : position
        */
        update = update.mul(mu).sub(gradient.mul(learningRate/(float)batchSize)); // update <- update * mu -  (learningRate/batchSize) * gradient
        value.addi(update); // value <- value + update
    }
}
