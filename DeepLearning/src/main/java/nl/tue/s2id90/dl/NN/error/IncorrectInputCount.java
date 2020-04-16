package nl.tue.s2id90.dl.NN.error;

/**
 *
 * @author Roel van Engelen
 */
public class IncorrectInputCount extends RuntimeException{
    
    /**
     * layer input / output Tensor shape mismatch
     * 
     * @param error_message 
     */
    public IncorrectInputCount( String error_message ){
        super( error_message );        
    }
}
