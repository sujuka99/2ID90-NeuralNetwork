package nl.tue.s2id90.dl.input;

import java.util.Arrays;
import java.util.Collections;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import nl.tue.s2id90.dl.Util;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Input_Reader
 * abstract data reader class
 * 
 * @author Roel van Engelen
 */
public abstract class InputReader{
   // thread safe lists
    @Getter @Setter(AccessLevel.PROTECTED) private List<TensorPair> trainingData;
    @Getter @Setter(AccessLevel.PROTECTED) private List<TensorPair> validationData;
    
    protected final int   batch_size;
    
    /**
     * 
     * @param batch_size 
     */
    public InputReader( int batch_size ){
        
        this.batch_size   = batch_size;
    }
    
    /**
     * Create training data iterator
     * 
     * @return training data iterator 
     */
    public Iterator<TensorPair> getTrainingBatchIterator(){
        
        // shuffle training data
        Collections.shuffle(getTrainingData() );
        
        // return batch iterator
        return new TrainingDataIterator( getTrainingData(), batch_size);
    }
    
    /**
     * Create training data iterator
     * 
     * @param n    number of batches of training  Data
     * @return training data iterator 
     */
    public Iterator<TensorPair> getTrainingBatchIterator(int n){
        
        if (n*batch_size>=getTrainingData().size()/2) {   
            // shuffle training data
            Collections.shuffle(getTrainingData() );

            // return batch iterator
            return new TrainingDataIterator( getTrainingData().subList(0, n*batch_size), batch_size);
        } else {
        
            List<TensorPair> subList = Util.pickRandom(trainingData, n*batch_size);
        
        // return batch iterator
        return new TrainingDataIterator( subList, batch_size);
        }
    }
    
    /**
     * Get amount of batches in training data
     * 
     * @return amount training data batches available per epoch
     */
    public int getTrainingBatchCount() {
        return getTrainingData().size() / batch_size;
    }
     
    /**
     * Get first #amount training pairs of validation data
     * 
     * @param amount
     * @return List with the first #amount validation data pairs
     */
    public List<TensorPair> getValidationData( int amount ){
        
        return InputReader.this.getValidationData().subList( 0, amount );
    }
    
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////// iterator
    
    /**
     * Training data iterator able to loop over all Tensor pairs and generate
     * correct list with batch_size tensor pairs
     */
    public class TrainingDataIterator implements Iterator<TensorPair> {
        
        private int sample_id;
        
        private final int batch_size;       
        private final List<TensorPair> data;
        protected final int[] input_shape;
        protected final int[] output_shape;
        protected final TensorShape input_tensor_shape;
        protected final TensorShape output_tensor_shape;
        private  INDArray input;
        private  INDArray output;
        
        /**
         * 
         * create iterator with data generating batch_size batches;
         * uses the input and output shape of Tensor_Pair data.get(0).
         * 
         * @param data
         * @param batch_size 
         */
        public TrainingDataIterator( List<TensorPair> data, int batch_size){
            this(data, batch_size,
                    stripFirst(data.get(0).model_input.getShape().getShape()),   // strip batch size from shape (typically, that should be 1)
                    stripFirst(data.get(0).model_output.getShape().getShape())
            );
        }

        /**
         * 
         * create iterator with data generating batch_size batches
         * 
         * @param data
         * @param batch_size 
         * @param input_shape
         * @param output_shape 
         */
        protected TrainingDataIterator( List<TensorPair> data, int batch_size, 
                                       int[] input_shape, int[] output_shape ){
                        
            this.sample_id  = 0;
            this.data       = data;
            this.batch_size = batch_size;
            this.input_shape  = create_shape_with_batch( batch_size, input_shape );
            this.output_shape = create_shape_with_batch( batch_size, output_shape );
            
            this.input_tensor_shape  = data.get( 0 ).model_input.getShape();
            this.output_tensor_shape = data.get( 0 ).model_output.getShape();
            
//            input  = Nd4j.create( input_shape, 'f' );
//            output = Nd4j.create( output_shape, 'f' );
        }

        /**
         * 
         * @return true  if another batch is available
         *         false if not
         */
        @Override
        public boolean hasNext() {
            
            return sample_id + batch_size <= data.size();       // Huub: replaced < by <= !
        }

        /**
         * Get net tensor pair batch
         * 
         * @return list with batch_size tensor pairs
         */
        @Override
        public TensorPair next(){
                        
            // to assure proper training these have to be initialized with f ordering
            INDArray input  = Nd4j.create( input_shape, 'f' );
            INDArray output = Nd4j.create( output_shape, 'f' );
            
            // add bach_size samples to batch INDArray
            for( int x = 0 ; x < batch_size ; x++ ){

                input.putRow( x, data.get( sample_id ).model_input.getValues() );
                output.putRow( x, data.get( sample_id ).model_output.getValues() );
                
                sample_id++;
            }
            
            // create input and output tensor
            Tensor in  = new Tensor( input,  input_tensor_shape  );
            Tensor out = new Tensor( output, output_tensor_shape );            
            
            // create tensor pair
            return new TensorPair( in, out );
        }
        
        /**
         * 
         * @param batch_size
         * @param shape
         * @return 
         */
        private int[] create_shape_with_batch( int batch_size, int[] shape ){
            
            int[] ret = new int[ shape.length + 1 ];
            
            ret[ 0 ] = batch_size;
            
            int x = 1;
            
            for( int dim : shape ){
                
                ret[ x ] = dim;
                x++;
            }
            
            return ret;
        }
    }
        
    static private int[] stripFirst(int[] a) {
        return Arrays.copyOfRange(a, 1, a.length);
    }
    
    public TensorShape getInputShape() {
        return getTrainingData().get(0).model_input.getShape();
    }
    
    public TensorShape getOutputShape() {
        return getTrainingData().get(0).model_output.getShape();
    }
    
    /** @return a map with named informational objects for this reader's data. */
    public Map<String,Object> getInfoMap() {
        Map result = new LinkedHashMap<>();
        result.put("reader class"      , this.getClass().getSimpleName());
        result.put("batch size"        , this.batch_size);
        result.put("#batches"          , this.getTrainingBatchCount());
        result.put("#training pairs"   , this.getTrainingData().size());
        result.put("#validation pairs", this.getValidationData().size());
        result.put("input shape"       , getInputShape().toString());
        result.put("output shape"      , getOutputShape().toString());
        return result;
    }
    
    @Override
    public String toString() {
        return Util.toString(getInfoMap());
    }
}
