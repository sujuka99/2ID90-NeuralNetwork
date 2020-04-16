package nl.tue.s2id90.dl.experiment;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import lombok.Getter;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.validate.ZeroValidator;
import nl.tue.s2id90.dl.input.InputReader;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * @author huub
 */
public class Experiment {   
    static final Logger LOGGER = Logger.getLogger(Experiment.class.getName());
    
    private BatchResult lastValidationResult;
    
    /** the number of epochs to perform in the experiment. This variable is set
     by the trainModel() method. 
     */
    @Getter int epochs;
    
    /** every activations number of batches the activation of the network
     * is made available to the gui; if less or equal to zero, this is ignored.
     * This variable is set by the trainModel() method. 
     */ 
    @Getter int activations;
    
    @Getter InputReader reader;
 
    public Experiment() {
    }
    
    /**
     * trains a neural network model, when finished saves the last model in the folder
     * "experiments/CLASSNAME-Model.json".
     * 
     * @param reader      the data source
     * @param sgd         the chosen optimizer, typically SGD
     * @param epochs      the number of epochs for training
     * @param activations every activations batches the activation of the network is made available to the gui;
     *                     if less or equal to zero, this is ignored.
     */
    public void trainAndSaveModel(Optimizer sgd, InputReader reader, int epochs, int activations) {
        try {
            trainModel(sgd, reader,epochs, activations);
        } finally {
            try {
                File file = Paths.get("experiments", getClass().getSimpleName()+"-Model.json").toFile();                
                LOGGER.log(Level.INFO, "Saving model to {0} ...", file.getCanonicalPath());
                sgd.getModel().saveModel(file);
            } catch (IOException ex) {
                LOGGER.severe(ex.getMessage());
            }
        }
    }

    /**
     * trains the neural network model.
     * @param reader      the data source
     * @param sgd         the chosen optimizer, typically SGD
     * @param epochs      the number of epochs for training
     * @param activations every activations batches the activation of the network is made available to the gui;
     *                     if less or equal to zero, this is ignored.
     */
    public void trainModel(Optimizer sgd, InputReader reader, int epochs, int activations) {
        this.epochs = epochs;
        this.activations = activations;
        this.reader = reader;
        
        Model model = sgd.getModel();
        // loop over all training batches for #epochs times
        for( int epoch = 1 ; epoch <= epochs ; epoch++ ){
            
            onEpochStart(sgd, epoch);
            
            // iterator randomizes data and loops over all training data
            Iterator<TensorPair> batches = reader.getTrainingBatchIterator();
            
            // loop over all training batches
            System.out.format( "\nTraining epoch %d ..,\n", epoch);
            
            // improved memory managment using nd4j's WorkSpaces
//            WorkspaceConfiguration learningConfig = WorkspaceConfiguration.builder()
//            .policyAllocation(AllocationPolicy.STRICT) // <-- this option disables overallocation behavior
//            .policyLearning(LearningPolicy.FIRST_LOOP) // <-- this option makes workspace learning after first loop
//            .build();
            
            // iterate over all batches
            while( batches.hasNext() ){
                
                // use the work space
//                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(learningConfig, "ONEBATCH")) {
                
                    TensorPair batch = batches.next();

                    // train model with one batch   
                    model.setInTrainingMode(true);    // ToDo: why is this in the loop?

                    onBatchStart(sgd,batch);

                    BatchResult result = sgd.trainOnBatch(batch);

                    onBatchFinished(sgd, batch, result);
//                }
            }
            
            onEpochFinished(sgd, epoch);
        }
        
        System.out.format( "Training of model finished after %d epochs.\n.", epochs );
    }
    
    /** called right before a batch is trained in method trainModel().
     * The arguments are the arguments as presented to trainModel, supplemented with
     * the last batch and its result.
     * @param sgd         the chosen optimizer, typically SGD.
     * @param batch       the last batch handled during training.
     */
    public void onBatchStart(Optimizer sgd, TensorPair batch) {
    }
    
    /** called after a batch has been trained in method trainModel().
     * The arguments are the arguments as presented to trainModel, supplemented with
     * the last batch and its result.
     * @param sgd         the chosen optimizer, typically SGD.
     * @param batch       the last batch handled during training.
     * @param result      the result of the last batch training.
     */
    public void onBatchFinished(Optimizer sgd, TensorPair batch, BatchResult result) {
        System.out.println(result);
    }
    
    /** called right before an epoch is trained in method trainModel().
     * @param sgd         the chosen optimizer, typically SGD.
     * @param epoch       index of the epoch that is about to start.
     */
    public void onEpochStart(Optimizer sgd, int epoch) {
    }
    
    /** called after an epoch has been trained in method trainModel.
     * The arguments are the arguments as presented to trainModel, supplemented with
     * the index of the last finished epoch..
     *
     * @param sgd         the chosen optimizer, typically SGD
     * @param epoch       index of the last finished epoch.
     */
    public void onEpochFinished(Optimizer sgd, int epoch) {
        if (sgd.getValidator() instanceof ZeroValidator) return;
        Model model = sgd.getModel();
        // validate model after each epoch
        System.out.println("\nValidating ...");
        model.setInTrainingMode(false);
        lastValidationResult = sgd.validate(reader.getValidationData());
        System.out.format("Validation after epoch %3d: %s \n", epoch, lastValidationResult);
    }
    
    /** returns the result of the last validation, or null if no validation was performed.
      *@return BatchResult 
      */
    public BatchResult getLastValidationResult() {
        return lastValidationResult;
    }
}