package nl.tue.s2id90.dl.NN.transform;

import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;


public class ScaleTransform implements DataTransform {
    private float sf;
    public ScaleTransform(float sf) {
        this.sf = sf;
    }
    @Override
    public void fit(List<TensorPair> data) {
    }

    @Override
    public void transform(List<TensorPair> data) {
        for(TensorPair pair: data) {
            pair.model_input.getValues().muli(sf);
        }
    }
    
}
