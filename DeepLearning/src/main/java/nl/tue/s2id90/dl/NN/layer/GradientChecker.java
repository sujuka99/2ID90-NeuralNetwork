package nl.tue.s2id90.dl.NN.layer;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.error.IllegalInput;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class GradientChecker {

    private final double EPS;
    private Model model;
    private final double DELTA;

    INDArray analyticGradient;

    public GradientChecker(Model model) {
        this(model, 1e-4, 1e-6);
    }

    public GradientChecker(Model model, double eps, double delta) {
        this.model = model;
        this.EPS = eps;
        this.DELTA = delta;
    }

    public void computeAnalyticGradient(TensorPair input) {
        analyticGradient = gradient(input);
    }

    public void check() {
        INDArray backpropGradient = getBackpropGradientFromModel();

        double maxNorm = backpropGradient.sub(analyticGradient).norm1Number().doubleValue();
        double relativeError = maxNorm/Math.max(
                backpropGradient.norm2Number().doubleValue(),
                analyticGradient.norm2Number().doubleValue()
        );
        
        System.err.println("\n\n>>> relative error: "+relativeError);

        if (relativeError > EPS) {
            double norm2gradient = analyticGradient.norm2Number().doubleValue();
            double norm2backprop = backpropGradient.norm2Number().doubleValue();
            INDArray p = analyticGradient.mmul(backpropGradient.transpose());
            System.err.println(Arrays.asList(Arrays.stream(p.shape()).boxed().collect(Collectors.toList())));

            double cosineSim = p.getDouble(0) / (norm2gradient * norm2backprop);
            double angle = Math.acos(cosineSim) * 180 / Math.PI;

            System.err.println(">>> cosineSim        = " + cosineSim);
            System.err.println(">>> angle           = " + angle);
            System.err.println(">>> Norm2   gradient=" + norm2gradient);
            System.err.println(">>> Norm2   backprop=" + norm2backprop);
            System.err.println(">>> maxNorm gradient=" + analyticGradient.normmaxNumber());
            System.err.println(">>> maxNorm backprop=" + backpropGradient.normmaxNumber());
            System.err.println(">>> maxNorm         =" + backpropGradient.sub(analyticGradient).normmaxNumber());
            INDArray a = backpropGradient.sub(analyticGradient);

            double max = Double.NEGATIVE_INFINITY;
            int maxIndex = -1;
            for (int i = 0; i < a.length(); i++) {
                double val = Math.abs(a.getDouble(i));
                if (val > max) {
                    maxIndex = i;
                    max = val;
                }
            }
            System.err.println(">>> maxIndex      =" + maxIndex);

            System.err.println(">>> max           =" + max);

            Layer layer = Layers.getLayerWithParameter(model, maxIndex).getKey();
            System.err.println(">>> maxLayer      =" + layer.getName());
            System.err.println(layer.getGradient());
        } else {
            System.err.println(">>> .");
        }
    }

    INDArray getBackpropGradientFromModel() {
        /**
         * concatenate all the gradients stored in the layers.
         */
        model.getLayers().stream()
                .filter(layer -> layer != null && layer.getGradient() != null)
                .forEach(layer
                        -> System.err.format("%s : %f ",
                        layer.getName(), layer.getGradient().normmaxNumber().doubleValue()
                )
                );

        // collect the gradients per layer in a list, ignoring parameterless layers.
        List<INDArray> list = model.getLayers().stream()
                .map(layer -> layer.getGradient())
                .filter(a -> a != null) // kick out results of layers without parameters
                .collect(Collectors.toList());

        // convert list to one big NDArray 
        int noParameters = (int)list.stream().mapToLong(a -> a.length()).sum();
        double[] buffer = new double[noParameters];
        INDArray[] arrays = list.toArray(new INDArray[0]);
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = Layers.getParameter(i, arrays);
        }
        return Nd4j.create(buffer);
    }

    /**
     * compute gradient component i. *
     */
    private double gradient_i(TensorPair input, int i) {
        // get correct labels
        Tensor labels = input.model_output;

        // store original parameter value
        final Double ORG_VALUE = Layers.getParameter(model, i);

        // compute loss for value+eps
        Layers.setParameter(model, i, ORG_VALUE + DELTA);
        model.inference(input.model_input);
        double loss2 = model.getOutputLayer().calculateLoss(labels);

        // compute loss for value-eps
        Layers.setParameter(model, i, ORG_VALUE - DELTA);
        model.inference(input.model_input);
        double loss1 = model.getOutputLayer().calculateLoss(labels);

        // restore original parameter value
        Layers.setParameter(model, i, ORG_VALUE);

        // return approximated derivative
        return (loss2 - loss1) / (2 * DELTA);
    }

    /**
     * compute full gradient.
     */
    private INDArray gradient(TensorPair input) {
        long batchSize = input.model_input.getValues().shape()[0];
        boolean mode = model.isInTrainingMode();
        model.setInTrainingMode(true);
        int n = Layers.getNumberOfParameters(model);
        double[] gradient = new double[n];

        for (int i = 0; i < n; i++) {
            gradient[i] = gradient_i(input, i)*batchSize;
        }
        model.setInTrainingMode(mode);
        return Nd4j.create(gradient);
    }

    public INDArray forwardAndBackwardPassWithoutUpdate(TensorPair input) throws IllegalInput {

        // set model in TrainingMode
        boolean mode = model.isInTrainingMode();
        model.setInTrainingMode(true);

        long batch_size = input.model_input.getValues().shape()[0];

        if (batch_size != 1) {
            throw new IllegalStateException("GradientChecker: batch-size should be 1");
        }

        // forward pass
        Tensor prediction = model.inference(input.model_input);

        // list with all layers
        List<Layer> layers = model.getLayers();

        // calculate backpropagation of output layer
        INDArray back = input.model_output.getValues();

        // loop over all layers from ( last -1 ) to first layer
        for (int x = layers.size() - 1; x > 0; x--) {

            back = layers.get(x).backpropagation(back);
        }

        // reset TrainingMode to original value
        model.setInTrainingMode(mode);

        return getBackpropGradientFromModel();
    }
}
