package nl.tue.s2id90.dl.input;

//import com.google.common.collect.Lists;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import java.util.ArrayList;
import java.util.Arrays;
import static java.util.Arrays.asList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Function;
import static java.util.stream.Collectors.toList;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Generate_Function_Data
 * generate input and output data from a list of Functions
 * 
 * @author Roel van Engelen
 */
public class GenerateFunctionData extends InputReader{        
    static final private Map<String,Function<Float,Float>>
            FMAP = new HashMap<>();
    static {
        FMAP.put("x^2",x -> (float) Math.pow ( x, 2.0 ));
        FMAP.put("x^3"    , x -> (float) Math.pow ( x, 3.0 ));
        FMAP.put("x^4"    , x -> (float) Math.pow ( x, 4.0 ));
        FMAP.put("abs(x)" , x -> (float) Math.abs ( x ));
        FMAP.put("sqrt(x)", x -> (float) Math.sqrt( x ));
        FMAP.put("exp(x)" , x -> (float) Math.exp ( x ));
        FMAP.put("x"      , x -> (float) x             );
    }
    private final TensorShape   shape_input;
    private final TensorShape   shape_output;
  
    // function list
    private final List<String> functions;
    
    public static GenerateFunctionData THREE_VALUED_FUNCTION(int batchSize) { 
        return THREE_VALUED_FUNCTION(1000,100,batchSize);
    }
    
    public static GenerateFunctionData THREE_VALUED_FUNCTION(int noTrainingSamples, int noValidationSamples, int batchSize) {
        // create list with functions to be used: input=x gives output=(x,x^2, e^x)
        List<String> fions = new ArrayList<>(asList("x", "x^2", "exp(x)"));

        // initialize image reader
        return new GenerateFunctionData(
            fions,                  // list with functions
            batchSize,              // batch size         
            1425,                   // random seed value
            noTrainingSamples,      // #training samples 
            noValidationSamples,    // #validation samples 
            -2,                     // training min input value
            2,                      // training max input value
            -1,                     // validation min input value
            1                       // validation max input value
        );
    }
    
    /**
     * generate input and output data from a list of Functions
     * 
     * @param functions            list with functions
     * @param batch_size           batch size  
     * @param seed                 random seed value
     * @param training_samples     amount of training samples
     * @param validation_samples   amount of validation samples
     * @param training_range_min   training min input value
     * @param training_range_max   training max input value
     * @param validation_range_min validation min input value
     * @param validation_range_max validation max input value
     */
    public GenerateFunctionData( List<String> functions, int batch_size, 
                                   long seed, int training_samples, int validation_samples, 
                                   int training_range_min, int training_range_max, 
                                   int validation_range_min, int validation_range_max ){
        super( batch_size );
                
        this.functions = functions;
        
        // input output shape
        shape_input  = new TensorShape( 1 );
        shape_output = new TensorShape( functions.size() );
        
        setTrainingData(new ArrayList<>());
        setValidationData(new ArrayList<>());
        generate_data( seed, training_samples, validation_samples, functions, 
                       training_range_min, training_range_max, 
                       validation_range_min, validation_range_max );
    }
    
    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private
    
    /**
     * 
     * @param seed
     * @param training_samples
     * @param validation_samples
     * @param functions
     * @param training_range_min
     * @param training_range_max
     * @param validation_range_min
     * @param validation_range_max 
     */
    private void generate_data( long seed, int training_samples, int validation_samples, List<String> functions, 
                                   float training_range_min, float training_range_max, 
                                   float validation_range_min, float validation_range_max ){
                
        HashSet<Float> training_sample_inputs   = new HashSet<>();
        HashSet<Float> validation_sample_inputs = new HashSet<>();
        
        float  value;
        Random random = new Random( seed );
        float  value_max = training_range_max - training_range_min;
        
        while( training_sample_inputs.size() < training_samples ){
            
            value = ( random.nextFloat() * value_max ) + training_range_min;
            training_sample_inputs.add( value );
        }
        
        value_max = validation_range_max - validation_range_min;
        
        while( validation_sample_inputs.size() < validation_samples ){
            
            value = ( random.nextFloat() * value_max ) + validation_range_min;
            if( !training_sample_inputs.contains( value ) ){
                
                validation_sample_inputs.add( value );
            }
        }
        
        generate_sample( getTrainingData(),   training_sample_inputs,   functions );
        generate_sample( getValidationData(), validation_sample_inputs, functions );
    }
    
    /**
     * 
     * @param list
     * @param inputs
     * @param functions 
     */
    private void generate_sample( List<TensorPair> list, HashSet<Float> inputs, List<String> fions ){
        List<Function<Float,Float>> fList = fions.stream().map(f->FMAP.get(f)).collect(toList());
        int[] nd4j_shape_input  = new int[]{ 1, 1 };
        int[] nd4j_shape_output = new int[]{ 1, fList.size() };
        
        for( float input : inputs ){
            
            // input tensor
            float[] values_input = new float[]{ input };
            Tensor tensor_input = new Tensor( Nd4j.create( values_input, nd4j_shape_input, 'c' ), shape_input );
            
            // output tensor
            float[] values_output = new float[ fList.size() ];
            for( int x = 0 ; x < fList.size(); x++ ){
                values_output[ x ] = fList.get( x ).apply( input );
            }        
            
            Tensor tensor_output = new Tensor( Nd4j.create( values_output, nd4j_shape_output, 'c' ), shape_output );
            
            list.add(new TensorPair( tensor_input, tensor_output ) );
        }        
    }
    
    @Override public Map<String,Object> getInfoMap() {
        List<String> headers = new ArrayList(functions);
        headers.add(0,"x");
        
        Map map = super.getInfoMap();
        map.put("headers",headers);
        return map;
    }
}
