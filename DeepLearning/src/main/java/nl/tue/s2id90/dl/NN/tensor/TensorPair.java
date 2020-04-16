package nl.tue.s2id90.dl.NN.tensor;

import static java.lang.String.format;
import java.util.Arrays;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Tensor_Pair
 * convenience class holding a Tensor pair, model input and model output
 * 
 * @author Roel van Engelen
 */
public class TensorPair {
    
    public final Tensor model_input;
    public final Tensor model_output;
    
    /**
     * Simple data class holding two tensors, model input and model output
     * 
     * @param model_input  model input  Tensor
     * @param model_output model output Tensor ( target output )
     */
    public TensorPair( Tensor model_input, Tensor model_output ){
        
        this.model_input  = model_input;
        this.model_output = model_output;
    }
    
    @Override
    public String toString() {
        return format("output=%s: input=%s",model_output,model_input,model_output);
    }
    
    
    /** combines the tensor pairs in the samples into a single TensorPair. **/
    static public TensorPair combine(List<TensorPair> samples) {
        int batchSize = samples.size();
        TensorPair firstSample = samples.get(0);
        int[] input_shape  = copyShapeAndSetBatchSize(batchSize,firstSample.model_input);
        int[] output_shape = copyShapeAndSetBatchSize(batchSize,firstSample.model_output);
            
        TensorShape input_tensor_shape  = samples.get( 0 ).model_input.getShape();
        TensorShape output_tensor_shape = samples.get( 0 ).model_output.getShape();
        
        return combine(samples,
                input_shape, output_shape, 
                input_tensor_shape, output_tensor_shape
        );
    }
    
    static private TensorPair combine(List<TensorPair> samples, int[] is, int[] os, TensorShape its, TensorShape ots) {

        // to assure proper training these have to be initialized with f ordering
        INDArray input  = Nd4j.create( is , 'f' );
        INDArray output = Nd4j.create( os, 'f' );
        
        int i=0;
        for(TensorPair s : samples) { 
                input.putRow(i, s.model_input.getValues() );
                output.putRow(i, s.model_output.getValues() );
                i=i+1;
        }

         // create input and output tensor
        Tensor in  = new Tensor( input,  its );
        Tensor out = new Tensor( output, ots );      

        return new TensorPair(in, out);
    }
    
    static private int[] copyShapeAndSetBatchSize(int batchSize, Tensor t) {
        int[] a = t.getShape().getShape();
        int[] b = Arrays.copyOfRange(a, 0, a.length);
        b[0]=batchSize;
        return b;
    }
}
