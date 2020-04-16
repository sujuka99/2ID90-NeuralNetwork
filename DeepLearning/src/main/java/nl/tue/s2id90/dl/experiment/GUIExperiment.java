package nl.tue.s2id90.dl.experiment;

import java.util.Map;
import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.scene.control.Accordion;
import javafx.scene.control.TitledPane;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.StopWatch;
import nl.tue.s2id90.dl.Util;
import nl.tue.s2id90.dl.javafx.FXGUI;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.javafx.Activations;
import nl.tue.s2id90.dl.javafx.GraphPanel;
import nl.tue.s2id90.dl.javafx.NodeUtil;
import nl.tue.s2id90.dl.javafx.widgets.VLabeledValueWidget;

/**
 * 
 * @author huub
 */
public class GUIExperiment extends Experiment {
    
    // the different types of visualizations
    private GraphPanel accuracyGraph;
    private GraphPanel lossGraph;
    private GraphPanel validationGraph;
    private Activations activationsPanel;
    private Accordion infoAccordion;
    
    // double values to be placed in top bar
    private double smoothLoss=1, noSamples=0, samplesPerSecond=0;
    private final DoubleProperty smoothLossProperty = new SimpleDoubleProperty(1);
    private final DoubleProperty noSamplesProperty = new SimpleDoubleProperty(0);
    private final DoubleProperty samplesPerSecondProperty = new SimpleDoubleProperty(0);
    
    // integer values to be placed in top bar
    int noBatches=0;
    private final IntegerProperty noBatchesProperty = new SimpleIntegerProperty(0);
    private final IntegerProperty epochIdProperty = new SimpleIntegerProperty(0);
    
    //stop watch
    private final StopWatch batchTime = new StopWatch(false); 
    private final StopWatch displayAttributesTime = new StopWatch(true);
    
    protected SimpleBooleanProperty doPlot= new SimpleBooleanProperty(true);
    
    public GUIExperiment() {
        // create javaFX widgets
        FXGUI fxGUI = FXGUI.getSingleton(); // initializes javafx platform!!
        fxGUI.setTitle(getClass().getSimpleName());
        
        // bind properties to widgets in top Bar
        VLabeledValueWidget smoothWidget = new VLabeledValueWidget("smooth loss","1"); 
        smoothWidget.valueProperty().bind(smoothLossProperty.asString("%e"));  
        
        VLabeledValueWidget noSamplesWidget = new VLabeledValueWidget("#samples","0");
        noSamplesWidget.valueProperty().bind(noSamplesProperty.asString("%e"));
        
        VLabeledValueWidget noBatchesWidget = new VLabeledValueWidget("#batches","0");
        noBatchesWidget.valueProperty().bind(noBatchesProperty.asString());
        
        VLabeledValueWidget millisWidget = new VLabeledValueWidget("#samples/sec","0");
        millisWidget.valueProperty().bind(samplesPerSecondProperty.multiply(1E9).asString("%e"));
        
        VLabeledValueWidget epochWidget = new VLabeledValueWidget("epoch","0");
        epochWidget.valueProperty().bind(epochIdProperty.asString());
        
        // fill top Bar
        FXGUI.getSingleton().addStatus(epochWidget, noBatchesWidget, noSamplesWidget, millisWidget, smoothWidget);
    }
    
    @Override
    public void trainAndSaveModel(Optimizer sgd, InputReader reader, int epochs, int activations) {
        addInfo("Data",reader.getInfoMap());
        addInfo("Model",sgd.getModel().getInfoMap());
        Map<String, Object> infoMap = sgd.getInfoMap();
        infoMap.put("epochs", epochs);
        infoMap.put("activations", activations);
        addInfo("Training", infoMap);
        super.trainAndSaveModel(sgd, reader, epochs, activations);
    }

    @Override
    public void trainModel(Optimizer sgd, InputReader reader, int epochs, int activations) {
        addInfo("Data", reader.getInfoMap());
        addInfo("Model", sgd.getModel().getInfoMap());
        Map<String, Object> infoMap = sgd.getInfoMap();
        infoMap.put("epochs", epochs);
        infoMap.put("activations", activations);
        addInfo("Training", infoMap);
        super.trainModel(sgd, reader, epochs, activations);
    }
    
    
    @Override
    public void onBatchStart(Optimizer sgd, TensorPair batch) {
        batchTime.start();
    }
    
    
    @Override
    public void onBatchFinished(Optimizer sgd, TensorPair batch, BatchResult result) {
        //super.onBatchFinished(sgd, batch, result);
        Model model = sgd.getModel();
        
        // add to gui
        if (doPlot.getValue()) {
            addTrainingResults(result);
        
            if (activations > 0 && result.batch_id % activations == 0) {
                model.setInTrainingMode(false);
                ActivationData activation = model.getActivations(batch, result.batch_id);
                addActivations(activation);
            }
        }
        
        // save values for usage in runLater method that is run in another thread.
        final long elapsedTime = batchTime.get();
        final long batchSize = batch.model_input.getValues().shape()[0];
        smoothLoss = 0.99*smoothLoss+0.01*result.loss;
        noSamples = noSamples + batchSize;
        noBatches = result.batch_id;
        samplesPerSecond = 0.99*samplesPerSecond + 0.01* (batchSize)/(double)elapsedTime;
        
        // update once per second
        updateStatusBar(displayAttributesTime.elapsedSecondsMoreThan(1));
    }
    
    @Override
    public void onEpochStart(Optimizer sgd, int epoch) {
        super.onEpochStart(sgd, epoch);
        Platform.runLater( () -> epochIdProperty.setValue(epoch));
    }
    
    @Override
    public void onEpochFinished(Optimizer sgd, int epoch) {
        super.onEpochFinished(sgd, epoch);
        if (doPlot.getValue()) addValidationResult(getLastValidationResult());
        
        // make sure at end of epoch, status is updated
        updateStatusBar(true);      
    }

    private void addTrainingResults(BatchResult result) {
        if (result.validation!=null) {
            if (accuracyGraph==null) {
                FXGUI.getSingleton().addTab(
                    accuracyGraph   = new GraphPanel("validation/batch")
                ); 
            }
            accuracyGraph.add(result.batch_id, result.validation);
        }
        
        if (lossGraph == null) {
            FXGUI.getSingleton().addTab(
                    lossGraph = new GraphPanel("loss/batch")
            );
        }
        lossGraph.add(result.batch_id, result.loss);
    }

    private void addValidationResult(BatchResult result) {
        if (result!=null && result.validation!=null) {
            if (validationGraph==null) {  
                FXGUI.getSingleton().addTab(
                    validationGraph = new GraphPanel("test set validation/epoch")
                );             
            }
            validationGraph.add(result.batch_id, result.validation);
        }
    }
    
    private void addActivations(ActivationData activation) {
        if (activationsPanel==null) {
            FXGUI.getSingleton().addTab(
                activationsPanel= new Activations()
            );
        }
        activationsPanel.add(activation);
    }
    
    public void addInfo(String label, Map<String,Object> infoMap) {
        Platform.runLater( () -> {
            if (infoAccordion==null)
                FXGUI.getSingleton().addTab(0, "info", infoAccordion = new Accordion());
            infoAccordion.getPanes().add(
                new TitledPane(label, NodeUtil.createInfoView(label,infoMap))
            );
            System.err.println(Util.toString(infoMap));
        }
        );
    }

    private void updateStatusBar(boolean condition) {
        if (condition)   {
            Platform.runLater(() -> {
                smoothLossProperty.setValue(smoothLoss);
                noSamplesProperty.setValue(noSamples);
                noBatchesProperty.setValue(noBatches);
                samplesPerSecondProperty.setValue(samplesPerSecond);
            });
            displayAttributesTime.start();
        }
    }
    
    public void setPlotting(boolean on) { doPlot.setValue(on); }
    public boolean isPlotting() { return doPlot.getValue(); }
}