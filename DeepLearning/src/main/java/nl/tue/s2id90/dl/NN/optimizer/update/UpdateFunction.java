package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Interface for update function to be used during training of the network.
 * @author huub
 */
public interface UpdateFunction {

    /** 
     * A typical implementation of this interface does the following
     *      array &lt;-- array - (learningRate/batchSize) * gradient.
     * However, other implementations may decide to ignore e.g. the
     * learningRate.
     * @param array         array that is to be updated, could contain either weights or biases.
     * @param isBias        true if and only if array represents bias values, as opposed to weights.
     * @param learningRate  learning rate for gradient descent
     * @param batchSize     the number of samples whose resulting gradients are accumulated in gradient
     * @param gradient      accumulated gradient
     */
    void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient);
    
    default String info() { return getClass().getSimpleName(); }
}
