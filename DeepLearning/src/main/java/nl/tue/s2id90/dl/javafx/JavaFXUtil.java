package nl.tue.s2id90.dl.javafx;

import java.util.List;
import javafx.application.Platform;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.input.ImageReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class JavaFXUtil {
        public static void computeNewImage(Model model, int width, int height, WritableImage wi) {
        int[] shape1x2= new int[]{width,2};
        INDArray data = Nd4j.create(new float[2*width], shape1x2, 'c');
        TensorShape shape2 = new TensorShape(2);
        Tensor t = new Tensor(data, shape2);
        
            
        PixelWriter pixelWriter = wi.getPixelWriter();
        PixelFormat format = PixelFormat.getByteRgbInstance();
        
        model.setInTrainingMode(false);
        for(int j=0;j<height;j++) {
            for(int i=0;i<width;i++) {
                data.putScalar(2*i, normalize(i, 0, width));
                data.putScalar(2*i+1, normalize(j,0,height));
            }
                
            INDArray rowValues = model.inference(t).getValues();  // [width x 3]
           
            
//            INDArray row = rowValues.reshape(1,3*width);          // 1 x (3*width)            
            // perform clamp255
//            BooleanIndexing.replaceWhere(row, 0, Conditions.lessThan(0));
//            BooleanIndexing.replaceWhere(row, 1, Conditions.greaterThan(1));
//            byte[] buffer = row.muli(255).data().asBytes();
               
            byte[] buffer = new byte[3*width];   
            for(int i=0;i<width;i++) {
                INDArray values = rowValues.getRow(i);
                buffer[3*i + 0] = clamp255(values.getDouble(0)+0.5);
                buffer[3*i + 1] = clamp255(values.getDouble(1)+0.5);
                buffer[3*i + 2] = clamp255(values.getDouble(2)+0.5);
            }
            
            int jj = j;
            Platform.runLater(() ->
                    pixelWriter.setPixels(0, jj, width, 1, format, buffer, 0, 0)
            );
        }
    }
    
    private static byte clamp255(double f) {
        if (f<0) f=0f; else if (f>1) f=1f;
        return (byte)(255*f);
    }
    
     public static float normalize(int i, int min, int max) {
        float average = (max-min)/2f;
        return (i-average)/(max-min);
    }
    
    private static float clamp(float f) {
        if (f<0) f =0f; else if (f>1) f=1f;
        return f;
    }
    
    private static byte clamp255(float f) {
        if (f<0) f=0f; else if (f>1) f=1f;
        return (byte)(255*f);
    }
    
    
    
    /** implemented for testing purposes; reproduces original image as expected. */
    public static void computeNewImage(ImageReader ir, int width, int height, WritableImage wi) {
            
        PixelWriter pixelWriter = wi.getPixelWriter();
        PixelFormat format = PixelFormat.getByteRgbInstance();
        
        for(int j=0;j<height;j++) {
            byte[] buffer = new byte[3*width];  
            for(int i=0;i<width;i++) {
                double[] rgb = ir.getRGB(i,j);
                buffer[3*i + 0] = clamp255(rgb[0]);
                buffer[3*i + 1] = clamp255(rgb[1]);
                buffer[3*i + 2] = clamp255(rgb[2]);
            }
            
            int jj = j;
            Platform.runLater(() ->
                    pixelWriter.setPixels(0, jj, width, 1, format, buffer, 0, 0)
            );
        }
    }
    
    /** implemented for testing purposes; reproduces original image from training data as expected. */
    public static void computeNewImage(List<TensorPair> trainingData, int width, int height, WritableImage wi) {    
        byte[] buffer = new byte[3*width*height];   // byte buffer for rgb values of full image
        for(TensorPair tp : trainingData) {
            INDArray ij = tp.model_input.getValues();
            int i = (int)Math.round((0.5+ij.getFloat(0))*width );
            int j = (int)Math.round((0.5+ij.getFloat(1))*height);
            INDArray rgb = tp.model_output.getValues();
            float r = rgb.getFloat(0);
            float g = rgb.getFloat(1);
            float b = rgb.getFloat(2);
            buffer[3*width*j+3*i+0] = clamp255(r);
            buffer[3*width*j+3*i+1] = clamp255(g);
            buffer[3*width*j+3*i+2] = clamp255(b); 
        }
        
        PixelFormat format = PixelFormat.getByteRgbInstance();
        Platform.runLater(() ->
            wi.getPixelWriter().setPixels(0, 0, width, height, format, buffer, 0, 3*width)
        );
    }
}
