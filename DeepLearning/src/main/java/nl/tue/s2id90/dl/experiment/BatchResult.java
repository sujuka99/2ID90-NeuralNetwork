package nl.tue.s2id90.dl.experiment;

import static java.lang.String.format;

/**
 * Batch_Result
 * Convenience class holding training/validation results, learning rate
 * loss for a single training/validation batch
 * 
 * @author Roel van Engelen
 */
public class BatchResult {
    
    public final int   batch_id;    
    public final Double loss;
    public final Float validation;
    public final Double learning_rate;
    
    /**
     * Initialize a Batch_Result with learning rate = 0 
     * 
     * @param batch_id current batch number
     * @param validation accuracy of current batch (may be null)
     */
    public BatchResult( int batch_id, Float validation){
        
        this.loss          = null;
        this.batch_id      = batch_id;
        this.validation    = validation;
        this.learning_rate = null;
    }
    
    /**
     * Initialize a Batch_Result
     * 
     * @param batch_id      current batch number
     * @param validation      validation of current batch (may be null)
     * @param loss          loss of current batch
     * @param learning_rate used learning rate 
     */
    public BatchResult( int batch_id, Float validation, double loss, double learning_rate ){
        
        this.loss          = loss;
        this.batch_id      = batch_id;
        this.validation      = validation;
        this.learning_rate = learning_rate;
    }
    
    /**
     * Print batch result data to system.out as
     * batch: [batch_id] accuracy: [accuracy] loss: [loss]
     * @return 
     */
    @Override
    public String toString(){
        if (loss!=null)
            return  format("(batch: %3d; validation: %4.8f;  loss: %4.8f)", batch_id, validation, loss);
        else
            return  format("(batch: %3d; validation: %4.8f)", batch_id, validation);
    }
    
    /**
     * Get specified data from current batch
     * 
     * @param graph_data requested data type
     * @return requested data or Float.MAX_VALUE if not available
     */
    public double get_data( GraphData graph_data ){
        
        // determine what data is requested and return it
        switch( graph_data ){
            case Training_Accuracy:
            case Validation_Accuracy:
                
                return validation;
            case Loss:
                
                return loss;
            case Learning_Rate:       
                
                return learning_rate;
        }
        
        return Float.MAX_VALUE;
    }
    
    /**
     * get batch id
     * 
     * @return batch_id
     */
    public float get_batch_id(){
        
        return batch_id;
    }
}
