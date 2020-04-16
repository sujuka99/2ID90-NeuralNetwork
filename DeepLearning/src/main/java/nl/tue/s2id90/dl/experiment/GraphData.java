package nl.tue.s2id90.dl.experiment;

/**
 * Graph_Data
 * Enum Typsetting all data stored in a Batch_Result
 * Loss, Learning_Rate, Training_Accuracy, Validation_Accuracy
 * 
 * @author Roel van Engelen
 */
public enum GraphData{
    
    Loss                ( "Loss" ),
    Learning_Rate       ( "Learning rate" ),
    Training_Accuracy   ( "Training Accuracy" ),
    Validation_Accuracy ( "Validation Accuracy" );
    
    private final String name;
    
    /**
     * Initialize enum with string name
     * 
     * @param name Graph data type name
     */
    private GraphData( String name ){
        
        this.name = name;
    }
    
    /**
     * get enum type name
     * 
     * @return Graph data type name
     */
    public String get_name(){
        
        return name;
    }
}
