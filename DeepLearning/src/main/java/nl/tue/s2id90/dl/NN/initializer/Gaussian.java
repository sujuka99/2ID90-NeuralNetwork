package nl.tue.s2id90.dl.NN.initializer;

import java.util.Random;
import java.util.function.BiFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class Gaussian implements Initializer{
        
    private final Random random;
    private BiFunction<Integer, Integer, Double> scale;
    
    /**
     * Create new Gaussian initiator
     */
    public Gaussian(){ 
        this( (fanIn,fanOut)->Math.sqrt(1.0/fanIn));
    }
    
    private Gaussian(BiFunction<Integer,Integer,Double> scale) {
        this.scale = scale; 
        // new random generator
        this.random = new Random();
    }
    
    /**
     * create flattened weights INDArray using a Gaussian distribution 
     * mean 0 and standard deviation determined by fanIn and/or fanOut.
     * @param fanIn
     * @param fanOut
     * @param shape weight shape
     * @return 
     */
    @Override
    public INDArray get_weight(int fanIn, int fanOut, int[] shape ){
        
        double std = scale.apply(fanIn, fanOut);
        double mean = 0.0;
        INDArray ret = Nd4j.rand(shape, Nd4j.getDistributions().createNormal(mean, std));
        
        INDArray flat = Nd4j.toFlattened('f', ret);        
        
        return flat;
    }
}
