package nl.tue.s2id90.dl.input;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.input.SampleDataGenerator;
import java.nio.ByteBuffer;
import java.util.Random;
import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritablePixelFormat;
import javafx.scene.paint.Color;
import javafx.util.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class PrimitivesDataGenerator extends SampleDataGenerator{
    final private static ImageGenerator GENERATOR= new ImageGenerator(
            true,         // square
            true,         // circle
            true,        // triangle
            true,        // rotated
            20180102,     // random seed
            28,           // image size
            0,            // noi,  not used if only nextImage() is called
            Color.BLACK,  // bgColor
            Color.WHITE,  // fgcolor
            false,         // anti-aliased
            null);        // Random generator (not used???)
    
    public PrimitivesDataGenerator(int batch_size, long seed, int training_samples, int validation_samples) {
        super(batch_size, seed, training_samples, validation_samples, 
            (Random rnd) -> { boolean firstCall = true;
                return toTensorPair(GENERATOR.nextImage());
            });
    }
    
    private static TensorPair toTensorPair(Pair<String,Image> pair) {
        String t=pair.getKey();
        Image image = pair.getValue();
        float[] input = { 
            (t.equals("S")?1.0f:0.0f),
            (t.equals("C")?1.0f:0.0f),
            (t.equals("T")?1.0f:0.0f)
        };
        INDArray classArray = Nd4j.create(input,new int[]{1,input.length},'c');
        INDArray imageArray = Nd4j.create(values(image),new int[]{1,1,28,28},'c');  // depth at the beginning
        return new TensorPair(
            new Tensor(imageArray, new TensorShape(28,28,1)),       // depth at the end
            new Tensor(classArray, new TensorShape(input.length))
        );
    }
    
    private static float[] values(Image image) {
//        int w = (int)image.getWidth(), h = (int)image.getHeight();
        int w=GENERATOR.getSize(), h=GENERATOR.getSize();
        byte[] bytes = new byte[4*w*h]; 
        PixelReader pr = image.getPixelReader(); 
        WritablePixelFormat<ByteBuffer> format = PixelFormat.getByteBgraInstance();
        pr.getPixels(0, 0, w, h, format, bytes, 0, 4*w); 
        float[] floatPixels = new float[w*h];
        for(int i=0;i<floatPixels.length;i++) {
            byte b = bytes[4*i], g=bytes[4*i+1], r=bytes[4*i+2], a=bytes[4*i+3];
            floatPixels[i]= (float)Color.rgb(r&0xFF, g&0xFF, b&0xFF).grayscale().getRed();
        }
        return floatPixels;
    }
}
