package nl.tue.s2id90.dl.NN.error;

/**
 *
 * @author Roel van Engelen
 */
public class IllegalLayer extends RuntimeException{
    
    /**
     * an illegal layer is used in the model
     * 
     * @param error_message 
     */
    public IllegalLayer( String error_message ){
        super( error_message );
        
    }
}