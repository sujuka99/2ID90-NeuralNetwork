package nl.tue.s2id90.dl.NN.layer;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import nl.tue.s2id90.dl.NN.activation.Activation;
import nl.tue.s2id90.dl.NN.loss.Loss;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.json.JSONUtil;
import static nl.tue.s2id90.dl.json.JSONUtil.getDouble;
import static nl.tue.s2id90.dl.json.JSONUtil.getInt;
import static nl.tue.s2id90.dl.json.JSONUtil.getString;
import org.json.simple.JSONObject;

/**
 *
 * @author huub
 */
public class JSONLayerFactory {
    public static Layer toLayer(JSONObject jo) {
        String type = getString(jo, "type");
        String name = getString(jo,"name");
        TensorShape input_shape  = fromJSON(jo,"input_shape",TensorShape::fromJson);
        TensorShape output_shape = fromJSON(jo,"output_shape",TensorShape::fromJson);
        switch(type) {
            case "InputLayer"    : {
                return new InputLayer(name, input_shape);
            }
            case "Convolution2D" : {
                Activation activation = fromJSON(jo,"activation",Activation::fromJson);
                int kernel_size = getInt(jo,"kernel_size");
                int kernels     = getInt(jo,"kernels");
                Convolution2D c2d = new Convolution2D(name, input_shape, kernel_size, kernels, activation);
                setWeights(jo,a->c2d.setWeights(a), b->c2d.setBias(b));
                return c2d;
            }
            case "Flatten"        : {
                return new Flatten(name,input_shape);
            }
            case "FullyConnected": {
                Activation activation = fromJSON(jo,"activation",Activation::fromJson);
                int outputs = output_shape.getShape(TensorShape.Dimension.SIZE);
                FullyConnected fc = new FullyConnected(name, input_shape, outputs, activation);
                setWeights(jo,a->fc.setWeights(a), b->fc.setBias(b));
                return fc;
            }
            case "OutputSoftmax" : {
                Loss loss = fromJSON(jo,"loss",Loss::fromJson);
                int classes = output_shape.getShape(TensorShape.Dimension.SIZE);
                OutputSoftmax op = new OutputSoftmax(name,input_shape,classes, loss);
                setWeights(jo,a->op.setWeights(a), b->op.setBias(b));
                return op;
            }
            case "PoolMax2D"    : {
                int stride = getInt(jo, "stride");
                return new PoolMax2D(name, input_shape, stride);
            }
            case "SimpleOutput"  : {
                Loss loss = fromJSON(jo,"loss",Loss::fromJson);
                int outputs = output_shape.getShape(TensorShape.Dimension.SIZE);
                SimpleOutput so = new SimpleOutput(name, input_shape, outputs, loss);
                setWeights(jo,a->so.setWeights(a), b->so.setBias(b));
            }
//            case "DropOutLayer"    : {
//                double probability = getDouble(jo, "probability");
//                return new DropOutLayer(name, input_shape, probability);
//            }
            default: throw new IllegalStateException("Unknownn layer type: "+type);
        }
    }
    
    private static void setWeights(JSONObject jo, Consumer<float[]> weights, Consumer<float[]> bias) {
        List<Double> wList = JSONUtil.toList(jo, "weights", Double.class);
        List<Double> bList = JSONUtil.toList(jo, "bias", Double.class);
        weights.accept(floats(wList));
        bias.accept(floats(bList));
    }
    
    private static <T> T fromJSON(JSONObject jo, String key, Function<JSONObject,T> f) {
        JSONObject job = (JSONObject)jo.get(key);
        return f.apply(job);
    }
    
    private static float[] floats(List<Double> list) {
        float[] result = new float[list.size()];
        int i=0; for(Double d: list) result[i++]=d.floatValue();
        return result;
    }
}
