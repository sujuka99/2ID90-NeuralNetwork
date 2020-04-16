package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;

/**
 *
 * @author huub
 */
public interface DataTransform {
    /** computes statistics for the dataset.
     * @param pairs **/
    void fit(List<TensorPair> pairs);
    
    /** transforms the dataset, using the statistics calculated by the fit method.
     * @param pairs **/
    void transform(List<TensorPair> pairs);
}
