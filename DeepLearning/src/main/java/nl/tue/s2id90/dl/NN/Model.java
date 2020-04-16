package nl.tue.s2id90.dl.NN;

import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.error.IllegalLayer;
import nl.tue.s2id90.dl.NN.error.IncorrectInputCount;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.Layer;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.experiment.ActivationData;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import static java.lang.String.format;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import static java.util.stream.Collectors.toList;
import java.util.stream.Stream;
import lombok.Getter;
import nl.tue.s2id90.dl.NN.layer.JSONLayerFactory;
import nl.tue.s2id90.dl.json.JSONUtil;
import nl.tue.s2id90.dl.json.JSONable;
import nl.tue.s2id90.dl.Util;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import nl.tue.s2id90.dl.NN.layer.OutputLayer;

/**
 * Model
 * 
 * @author Roel van Engelen
 * 
 */
public class Model implements JSONable {
    
    @Getter private final List<Layer>  layers;
    @Getter private       OutputLayer outputLayer = null;
    
    /**
     * initialize new model with input layer
     * 
     * @param input a Input layer as the first layer of the network
     */
    public Model( InputLayer input ){
        
        layers = new ArrayList<>();
        layers.add( input );
    }
    
    ////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////// public
    
    /**
     * add new layer to model. Throws an exception if a layer is added after an output 
     * layer has been added.
     * 
     * @param new_layer layer to add to model
     * @throws nl.tue.s2id90.dl.NN.error.IllegalLayer
     */
    public void addLayer( Layer new_layer ) throws IllegalLayer {
                
        if( outputLayer != null ){
            
            throw new IllegalLayer( "You may not add another layer to the model after an output layer has been added." );
        }
        if (new_layer instanceof OutputLayer) {
            this.outputLayer = (OutputLayer)new_layer;
        }
        layers.add( new_layer );
    }
    
    /**
     * Initialize all layer bias and weights and
     * check all input output shapes
     * 
     * @param initializer an initializer used to initialize bias and weights
     * @throws nl.tue.s2id90.dl.NN.error.IncorrectInputCount
     */
    public void initialize( Initializer initializer ) throws IncorrectInputCount{
        
        // get first layer output shape
        TensorShape input = layers.get( 0 ).getOutputShape();
        
        // loop over all layers after the first layer and
        // verify that connected output -> input shapes are equal
        for( int layer_id = 1 ; layer_id < layers.size() ; layer_id++ ){
            
            // verify that output shape from previous layer is the same as
            // input layer from this layer
            if( !input.isCorrectShape( layers.get( layer_id ).getInputShape().getShape() ) ){
                
                // shapes not equal, throw informative error
                throw new IncorrectInputCount( "Layer: " + layers.get( layer_id ).getName() +
                           " expected: " + layers.get( layer_id ).getInputShape().shapeToString() +
                           " found: " + input.shapeToString() );
            }
            
            // new input shape is output shape
            input = layers.get( layer_id ).getOutputShape();
        }
        
        // initialize all weights and biases 
        initializer.initialize_layers( layers );
    }
    
    /**
     * Calculate a prediction with the given input tensor
     * 
     * @param input a tensor that matches input layer input shape
     * @return Tensor with model prediction
     * @throws nl.tue.s2id90.dl.NN.error.IllegalInput
     */
    public Tensor inference( Tensor input ) throws IllegalInput{
        
        Tensor output = input;
        
        // loop over all layers in network
        for( Layer layer : layers ){
            
            // calculate inference layer with output from previous layer
            output = layer.inference( output );
        }
        
        return output;
    }
    
    /**
     * Calculate inference with given input and generate activation images
     * for all layers
     * 
     * @param pair input&amp;label Tensor pair
     * @param batch_id current batch id
     * @return
     * @throws IllegalInput 
     * ToDo: why is this a method of Model?????
     */
    public ActivationData getActivations( TensorPair pair, int batch_id ) throws IllegalInput{
        
        if (isInTrainingMode()) {
            throw new IllegalStateException("Model validated while in training Mode");
        }
        
        Tensor output = pair.model_input;
        // list with layer names
        List<String> names         = new ArrayList<>();
        // list with activation tensors
        List<Tensor> tensors       = new ArrayList<>();
        // list with show_values settings for every layer
        List<Boolean> show_values  = new ArrayList<>();
        
        // loop over all layers in network
        for( Layer layer : layers ){
            
            // calculate inference layer with output from previous layer
            output = layer.inference( output );
            
            // add layer name
            names.add( layer.getName() + " - shape: " + layer.getOutputShape().shapeToString() );
            // add activation tensor
            tensors.add( output.getCopy() );
            // add show values setting
            show_values.add( layer.showValues() );
        }
        
        // add label name
        names.add( "Label" );
        // add show values setting
        show_values.add( layers.get( layers.size() - 1 ).showValues() );
        // add activation tensor
        tensors.add( pair.model_output.getCopy() );
        
        return new ActivationData( batch_id, names, tensors, show_values );
    }    
    
    /**
     * Save model to file
     * 
     * @param file file location to store model
     * @throws java.io.FileNotFoundException
     */
    public void saveModel( File file ) throws FileNotFoundException{
        try (PrintWriter pw = new PrintWriter(file)) {
            pw.append(JSONUtil.format(json().toJSONString()));
        }
    }
    
    /**
     * Load model to file
     * 
     * @param file file location to load model from.
     * @return 
     * @throws java.io.FileNotFoundException 
     * @throws org.json.simple.parser.ParseException 
     */
    public static Model loadModel( File file ) throws FileNotFoundException, ParseException {
        return fromJson(new Scanner(file).useDelimiter("\\Z").next());
    }
    
    /**
     * Sets trainingMode in all layers of this model.
     * @param on
     */
    public void setInTrainingMode(boolean on) {
        // loop over all layers in network
        for( Layer layer : layers ){
            layer.setInTrainingMode(on);
        }
    }
    
     /**
     * 
     * @return whether or not all layers are in training mode 
     */
    public boolean isInTrainingMode() {
        return layers.stream().allMatch(layer->layer.isInTrainingMode());
    }

    @Override
    public JSONObject json() {
        JSONObject jo = new JSONObject();
        jo.put("layers", layers.stream()
                .map(l->l.json())
                .collect(Collectors.toList())
        );
        return jo;
    }
    
     public static Model fromJson(String json) throws ParseException {
        JSONObject jo = (JSONObject)new JSONParser().parse(json);
        
        JSONArray array = (JSONArray)jo.get("layers");
        
        Stream<JSONObject> stream = array.stream()
                .map(obj->(JSONObject)obj);
        
        List<Layer> layers = stream.map(obj->JSONLayerFactory.toLayer(obj)).collect(toList()); 
        
        Model model = new Model((InputLayer)layers.get(0));
        
        for(int i=1;i<layers.size();i++) {
            try {
                model.addLayer(layers.get(i));
            } catch (IllegalLayer ex) {
                Logger.getLogger(JSONUtil.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return model;
    }
     
     /** @return a map with named informational objects for this model. */
    public Map<String,Object> getInfoMap() {
        Map result = new LinkedHashMap<>();
        int i=1;
        for (Layer layer : layers) {
            String key = layer.getClass().getSimpleName();
            result.put(format("(%d) %s",i++,key),layer.getInfoMap());
        }
        return result;
    }
    
    @Override
    public String toString() {
        return Util.toString(getInfoMap());
    }
}
