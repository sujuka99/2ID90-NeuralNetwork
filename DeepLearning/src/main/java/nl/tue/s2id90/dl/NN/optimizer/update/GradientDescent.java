package nl.tue.s2id90.dl.NN.optimizer.update;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class GradientDescent implements UpdateFunction {
    /**
     * Does a gradient descent step with factor minus learningRate and corrected for batchSize.
     * @param value
     * @param isBias
     * @param gradient
     */
    @Override
    public void update(INDArray value, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
        double factor = -(learningRate/batchSize);
        Nd4j.getBlasWrapper().level1().axpy( value.length(), factor, gradient, value );
                                            // value <-- value + factor * gradient
    }
}
