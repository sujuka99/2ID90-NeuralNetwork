package nl.tue.s2id90.dl.NN.error;

/**
 *
 * @author Roel van Engelen
 */
public class IllegalInput extends RuntimeException{
    
    /**
     * input Tensor has incorrect shape
     * @param error_message 
     */
    public IllegalInput( String error_message ){
        super( error_message );
        
    }
}