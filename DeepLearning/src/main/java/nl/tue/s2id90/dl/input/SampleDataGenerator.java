package nl.tue.s2id90.dl.input;

import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * generates input and output data.
 * 
 * @author Huub van de Wetering
 */
public class SampleDataGenerator extends InputReader{
    
    // function to generate one sample, given the random seed generator
    private final Function<Random, TensorPair> generateOne;
    
    /**
     * generate input and output data from a list of Functions
     * 
     * @param batch_size           batch size  
     * @param seed                 random seed value
     * @param training_samples     amount of training samples
     * @param validation_samples   amount of validation samples
     * @param generateOne          function to generate one sample
     */
    public SampleDataGenerator(int batch_size, 
                long seed, int training_samples, int validation_samples,
                Function<Random,TensorPair> generateOne
    ){
        super( batch_size );
        
        this.generateOne = generateOne;
        
        setTrainingData(new ArrayList<>());
        setValidationData(new ArrayList<>());
        generate_data( seed, training_samples, validation_samples);
    }
    
    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private
    
    /**
     * 
     * @param seed
     * @param training_samples
     * @param validation_samples
     */
    private void generate_data( long seed, int training_samples, int validation_samples){
        Random random = new Random( seed );
        List<TensorPair> training=getTrainingData();
        List<TensorPair> validation=getValidationData();
        
        while( training.size() < training_samples ){
            
            TensorPair value = generateOne.apply(random);
            training.add( value );
        }
        
        while( validation.size() < validation_samples ){
            
            TensorPair value = generateOne.apply(random);
            if( !validation.contains( value ) ){
                
                validation.add( value );
            }
        }
    }
}