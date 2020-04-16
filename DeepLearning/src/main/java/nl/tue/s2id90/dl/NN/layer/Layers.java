package nl.tue.s2id90.dl.NN.layer;

import java.util.Arrays;
import javafx.util.Pair;
import nl.tue.s2id90.dl.NN.Model;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author huub
 */
class Layers {
    public static Pair<Layer,Integer> getLayerWithParameter(Model model, int i) {
        for(Layer layer : model.getLayers()) {
            Double d=layer.getParameter(i);
            if (d!=null) return new Pair(layer,i);
            i = i - layer.getNumberOfParameters();
        }
        return null;
    }
    
    public static Double getParameter(Model model, int i) {
        Pair<Layer,Integer> p = getLayerWithParameter(model,i);
        return p==null? null : p.getKey().getParameter(p.getValue());
    }
    
    public static boolean setParameter(Model model, int i, double d) {
        Pair<Layer,Integer> p = getLayerWithParameter(model,i);
        return p==null ? false : p.getKey().setParameter(p.getValue(), d);
    }
    
    public static int getNumberOfParameters(Model model) {
        return model.getLayers().stream()
                    .mapToInt(layer -> layer.getNumberOfParameters())
                    .sum();
    }
    
    /** returns the total number of elements in the given arrays. **/
    public static int getNumberOfParameters(INDArray... arrays) {
        return (int)Arrays.stream(arrays).mapToLong(array->array.length()).sum();
    }
    
    /** returns the i-th element in the array obtained by concatenating the given arrays.
     * 
     * @param i    index of element that has to be returned
     * @param arrays  array of INDArrays
     * @return the value of element i, null if that does not exist.
     */
    public static Double getParameter(long i, INDArray... arrays) {  
        for (INDArray a : arrays) {
            if (i<a.length()) {
                return a.getDouble(i);
            } else {
                i = i -a.length();
            }
        }
        return null;
    }
    
     /** sets the i-th element in the array obtained by concatenating the given arrays.
     * 
     * @param i    index of element that has to be set
     * @param arrays  array of INDArrays
     * @return true, on success, false otherwise
     */
    public static boolean setParameter(long i, double d, INDArray... arrays) {  
        for (INDArray a : arrays) {
            if (i<a.length()) {
                a.putScalar(i,d);
                return true;
            } else {
                i = i -a.length();
            }
        }
        return false;
    }
    
    public static void clearParameters(INDArray ... arrays) {
        for(INDArray a : arrays) {
            a.assign(0);
        }
    }
}
