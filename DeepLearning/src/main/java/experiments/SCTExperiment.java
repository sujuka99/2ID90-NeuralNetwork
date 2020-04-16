package experiments;

import java.io.IOException;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.activation.RELU;
import nl.tue.s2id90.dl.NN.initializer.Gaussian;
import nl.tue.s2id90.dl.NN.layer.Convolution2D;
import nl.tue.s2id90.dl.NN.layer.Flatten;
import nl.tue.s2id90.dl.NN.layer.InputLayer;
import nl.tue.s2id90.dl.NN.layer.OutputSoftmax;
import nl.tue.s2id90.dl.NN.layer.PoolMax2D;
import nl.tue.s2id90.dl.NN.loss.CrossEntropy;
import nl.tue.s2id90.dl.NN.optimizer.Optimizer;
import nl.tue.s2id90.dl.NN.optimizer.SGD;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.transform.DataTransform;
import nl.tue.s2id90.dl.NN.validate.Classification;
import nl.tue.s2id90.dl.experiment.GUIExperiment;
import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.input.MNISTReader;
import nl.tue.s2id90.dl.javafx.ShowCase;


public class SCTExperiment extends GUIExperiment {
    int batchSize = 64;
    int epochs = 10;
    double learningRate = 0.05;
    
    String[] labels = {
        "triangle", "square", "circle" 
    };
    ShowCase showCase = new ShowCase(i -> labels[i]);
    
    public void go() throws IOException {
        // read input and pr int some informat ion on the data
        InputReader reader = MNISTReader.primitives(batchSize);
        System.out.println("Reader info :\n" + reader.toString());
       
        //implementing mean subtraction on the current input
        DataTransform ms = new MeanSubtraction();
        
        ms.fit(reader.getTrainingData());
        ms.transform(reader.getTrainingData());
        ms.transform(reader.getValidationData());
        
        
        /*Initialize inputs and outputs, inputs is of type TensorShape since
        the constructor of Flatten requires parameter of this type
        */
        TensorShape inputs = reader.getInputShape();
        int outputs = reader.getOutputShape().getNeuronCount();
        
        
        //create model and initialize weights
        Model model = createModel(inputs, outputs);
        model.initialize(new Gaussian());
        System.out.println(model);//print the model
        
        Optimizer sgd = SGD.builder()
            .model(model)
            .learningRate(learningRate)
            .validator(new Classification())
            //.updateFunction(GradientDescentMomentum::new)//add gradient descent with momentum
            .updateFunction(() -> new L2Decay(GradientDescentMomentum::new,0.0001f))
            .build();
        trainModel(sgd, reader, epochs, 0);
    }
    
    public static void main (String [] args) throws IOException {
        new ZalandoExperiment().go();
    }
    
     Model createModel(TensorShape inputs, int outputs){
        int kernel1 = 5;//kernel size
        int stride = 2; //typical pooling stride
        
        Model model = new Model(new InputLayer("In", inputs, true));
        //add convolution layer
        model.addLayer(new Convolution2D("kernel1",inputs, kernel1, 1 ,new RELU()));
        //add pooling layer
        model.addLayer(new PoolMax2D("Pooling", inputs, stride));
        //flatten layer before output
        model.addLayer(new Flatten("Flatten", inputs));
        //add output layer with Softmax activation function and crossentropy 
        //loss fucntion
        model.addLayer(new OutputSoftmax("Out", new TensorShape(inputs.getNeuronCount()), outputs, new CrossEntropy()));
//        System.out.println(model);
        return model;
    }
     
    @Override
    public void onEpochFinished(Optimizer sgd, int epoch) {
        super.onEpochFinished(sgd, epoch);
        showCase.update(sgd.getModel());
    }
}
