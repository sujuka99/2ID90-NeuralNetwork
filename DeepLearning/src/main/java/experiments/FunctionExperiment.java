package experiments;
import nl.tue.s2id90.dl.experiment.Experiment;
import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.FullyConnected;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.SimpleOutput;
import nl.tue.s2id90.dl.NN.loss.MSE;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.validate.Regression;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.GenerateFunctionData;
import nl.tue.s2id90.dl.input.InputReader;

public class FunctionExperiment extends GUIExperiment{
    //(hyper) parameter
    //...
    int batchSize = 16;//100
    int epochs = 70;//13
    double learningRate = 0.1;
    
    public void go() throws IOException {
        InputReader reader = GenerateFunctionData.THREE_VALUED_FUNCTION(batchSize);
        System.out.println("Reader info :\n" + reader.toString());
        
        int inputs = reader.getInputShape().getNeuronCount();
        int outputs = reader.getOutputShape().getNeuronCount();
        
        Model model = createModel(inputs, outputs);
        model.initialize(new Gaussian());
        
        // Training: create and configure SGD&&train model
        Optimizer sgd = SGD.builder()
            .model(model)
            .validator(new Regression())
            .learningRate(learningRate)
            .build();
        trainModel(sgd, reader, epochs, 0);
    }
    
    public static void main (String [] args ) throws IOException {
        new FunctionExperiment().go();
    }
    
    Model createModel(int inputs, int outputs){
        Model model = new Model(new InputLayer ("In", new TensorShape(inputs), true));
        //next we add one hidden layer fc1
        model.addLayer(new FullyConnected("fc1", new TensorShape(inputs), 16, new RELU()));
        model.addLayer(new SimpleOutput("Out", new TensorShape(16), outputs, new MSE(), true));
        return model;
    }
}
