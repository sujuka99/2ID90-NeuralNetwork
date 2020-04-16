package nl.tue.s2id90.dl.NN.loss;

import nl.tue.s2id90.dl.NN.activation.Activation;
import nl.tue.s2id90.dl.json.JSONable;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Roel van Engelen
 */
public interface Loss extends JSONable {
    
    /**
     * Calculate loss over label and pre activation
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @param activation  ND4J Iactivation type
     * @return            cross entropy loss value
     */
    public double calculate_loss( INDArray labels, INDArray preoutput, Activation activation );
    
    
    /**
     * calculate final layer backpropagation gradient
     * 
     * @param labels      tensor with correct output
     * @param preoutput   tensor with pre activation output
     * @param activation  ND4J Iactivation type
     * @return            INDArray backpropagation gradient
     */
    public INDArray computeGradient( INDArray labels, INDArray preoutput, Activation activation );
    
     public static Loss fromJson(JSONObject jo) {
        String type = (String)jo.get("type");
        switch(type) {
            case "CrossEntropy": return new CrossEntropy();
            case "MSE": return new MSE();
            case "L1": return new L1();
            default: throw new IllegalStateException("Unknownn loss type: "+type);
        }
     }
}
