package nl.tue.s2id90.dl.input;

import java.awt.image.BufferedImage;
import java.io.File;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javafx.scene.image.Image;
import javafx.scene.paint.Color;
import javax.imageio.ImageIO;
import lombok.Getter;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 
 * read CIFAR10 data from file and create training and validation tensors.
 * 
 * @author Huub van de Wetering
 */
public class ImageReader extends InputReader{
    final private int WIDTH;
    final private int HEIGHT;
    final private int DEPTH  = 3;
//    @Getter final private BufferedImage image;
    @Getter final private Image fxImage;
    
    
    
    /**
     * Read all CIFAR10 images of all possible classes
     * 
     * @param file   image file (.jpg, .png)
     * @param batch_size amount of training data pairs in one batch
     * @throws java.io.IOException 
     */
    public ImageReader(File file, int batch_size) throws IOException{
        super( batch_size);
//        image = readImage(file);
        fxImage = readFXImage(file);
        WIDTH = (int)fxImage.getWidth();
        HEIGHT = (int)fxImage.getHeight();
        setTrainingData(readData());
        setValidationData(new ArrayList<>());
    }
    
    /** returns Buffered Image of file, null if file can not be read. */
    private BufferedImage readImage(File file) {
        try {
            return ImageIO.read(file);
        } catch(IOException e) {
            return null;
        }
    }
    
    /** returns Buffered Image of file, null if file can not be read. */
    private Image readFXImage(File file) {
        try {
            return new Image(file.toURI().toURL().toExternalForm(),200d,200d,false,true);
        } catch(IOException e) {
            return null;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private 

    private List<TensorPair> readData() {
        List<TensorPair> result = new ArrayList<>(WIDTH*HEIGHT); // we know the number of pixels in the image

        int[] xyShape = new int[]{1, 2};        //
        int[] rgbShape = new int[]{1, DEPTH};
        for (int i = 0; i < WIDTH; i++) {
            for (int j = 0; j < WIDTH; j++) {
                // construct input Tensor
                TensorShape ts0 = new TensorShape(2);
                double[] xy = new double[]{normalize(i,0,WIDTH),normalize(j,0,HEIGHT)};
                Tensor t0 = new Tensor(Nd4j.create(xy, xyShape,'c'), ts0);
                
                // construct output Tensor
                TensorShape ts1 = new TensorShape(DEPTH);// read image and create input tensor
                double[] rgb = getRGB(i,j);
                Tensor t1 = new Tensor(Nd4j.create(rgb, rgbShape, 'c'), ts1);

                result.add(new TensorPair(t0, t1));
            }
        }
        return result;
    }
    
    private double normalize(double i, int min, int max) {
        float average = (max-min)/2f;
        return (i-average)/(max-min);
    }

    /** returns rgb of pixel (i,j) of image */
    public double[] getRGB(int i, int j) {
        Color color = fxImage.getPixelReader().getColor(i, j);
        return new double[] {
            normalize(color.getRed  (),0,1),
            normalize(color.getGreen(),0,1),
            normalize(color.getBlue (),0,1)
        };
    }
    
    /** returns hsv of pixel (i,j) of image */
    public double[] getHSV(int i, int j) {
        Color color = fxImage.getPixelReader().getColor(i, j);
        return new double[] {color.getHue()/360.0f,color.getSaturation(),color.getBrightness()};
    }
}