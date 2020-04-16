package nl.tue.s2id90.dl.input;

import nl.tue.s2id90.dl.input.InputReader;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import static java.lang.String.format;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import static java.util.stream.Collectors.toList;
import javafx.scene.paint.Color;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.Nd4jUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IAMax;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * 
 * read CIFAR10 data from file and create training and validation tensors.
 * 
 * @author Huub van de Wetering
 */
public class Cifar10Reader extends InputReader{
    
    // thread safe lists
    private  List<TensorPair> original_data_training;
    private  List<TensorPair> original_data_validation;

    final private int WIDTH  = 32;
    final private int HEIGHT = 32;
    final private int DEPTH  = 3;
    final private int NOCLASSES;
    
    
    
    /**
     * Read all CIFAR10 images of all possible classes
     * 
     * @param batch_size amount of training data pairs in one batch
     * @throws java.io.IOException 
     */
    public Cifar10Reader( int batch_size) throws IOException{
        this( batch_size, 10 );
    }
    
    /**
     * Read all CIFAR10 images of classes 0 .. noClasses
     * 
     * @param batch_size amount of training data pairs in one batch
     * @param noClasses     number of different classes read ( at most 10)
     * @throws java.io.IOException  
     */
    public Cifar10Reader( int batch_size, int noClasses ) throws IOException{
        super( batch_size );
                        
        // MNIST training label and image file locations
        final String BASE = "data/cifar-10-batches-bin/";
        final String PREFIX = "data_batch_";
        String batch1 = BASE + PREFIX+"1"+".bin";
        String batch2 = BASE + PREFIX+"2"+".bin";
        String batch3 = BASE + PREFIX+"3"+".bin";
        String batch4 = BASE + PREFIX+"4"+".bin";
        String batch5 = BASE + PREFIX+"5"+".bin";
        List<String> trainingBatches = Arrays.asList(batch1, batch2, batch3, batch4, batch5);
        List<String> testBatches    = Arrays.asList(BASE  + "test_batch.bin");
        
        // read training and validation data
        this.NOCLASSES = Math.min(noClasses,10);      // no more than 10 classes possible
        setTrainingData(original_data_training = readData(trainingBatches));
        setValidationData(original_data_validation = readData(testBatches));
    }
    
    public static  List<String> getLabelsAsString() {
        return Arrays.asList(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        );
    }

    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private 
    
    private List<TensorPair> readData(List<String> batches) {
        return batches.stream()
                .flatMap(batch->readData(batch).stream())
                .filter(tp-> indexOfOne(tp.model_output.getValues()) < NOCLASSES)
                .collect(toList());
    }
    
    private int indexOfOne(INDArray a) {
        return Nd4j.getExecutioner().execAndReturn(new IAMax(a)).getFinalResult();
    }

    private List<TensorPair> readData(String batch) {
        List<TensorPair> result = new ArrayList<>(10000); // we know that there are 10000 images in one batch
        
        int[]        indar_label = new int[]{ NOCLASSES };
        int[]        indar_image = new int[]{ 1, DEPTH, WIDTH, HEIGHT };
        
        try (FileInputStream is = new FileInputStream(batch)) {
            for(int i=0;i<10000;i++) {
                // read label, target output, and output tensor
                byte[] label = new byte[1   ];
                is.read(label);
                
                if (label[0]<NOCLASSES) {
                    float[] labelf = new float[NOCLASSES];
                    labelf[ label[0] ] = 1;
                    TensorShape ts1 = new TensorShape(NOCLASSES);                
                    Tensor t1 = new Tensor( Nd4j.create( labelf, indar_label, 'f' ), ts1 );  

                    // read image and create input tensor
                    float[] rgb = read(is,3*1024);
//                    float[] hsv = toHSV(rgb);
                    TensorShape ts0 = new TensorShape(WIDTH,HEIGHT,DEPTH);// create label tensor
                    Tensor t0 = new Tensor( Nd4j.create( rgb, indar_image, 'c' ), ts0 );
                    
                    result.add(new TensorPair(t0,t1));
                } else {
                    // just read the image from is, and forget about it
                    read(is,3*1024);
                }
            }
        } catch (IOException e) {
                System.err.println(format("error reading %s\n%s", batch, e));
        }
        return result;
    }
    
    float[] toHSV(float[] rgb) {
        float[] hsv = new float[3*1024];
        for(int i=0;i<rgb.length/3;i++) {
            double r = rgb[i], g=rgb[1024+i], b=rgb[2048+i];
            Color c = Color.rgb((int)(r),(int)(g), (int)(b));
            hsv[     i] = (float)c.getHue();
            hsv[1024+i] = (float)c.getSaturation();
            hsv[2048+i] = (float)c.getBrightness();
        }
        return hsv;
    }
    
    /** returns array of floats of length size, by first reading size bytes, and
     * converting each byte in to a float in range [0,255].
     */
    private float[] read(InputStream is, int size) throws IOException {
        byte[] ba = new byte[size];
        is.read(ba);
        return floatArray(ba);
    }
    
    /** returns same size array where each byte is converted in to a float in the range [0,255] */
    private float[] floatArray(byte[] b) {
        float[] f = new float[b.length];
        for(int i=0;i<b.length;i++) {
            f[i]=(float)(b[i]&0xFF);       // float in [0,255]
        }
        return f;
    }
    
    public void augmentData(int ulx, int uly, int widthx, int widthy, boolean training) {
        if (training) {
            setTrainingData(cropAndOrFlip(ulx, uly, widthx, widthy, original_data_training, 0.5));
        } else {  // crop validation data: typically, this takes the center part of the image; No flipping!
            setValidationData(cropAndOrFlip(ulx, uly, widthx, widthy, original_data_validation, 0.5d));
        }
    }
    
    public List<TensorPair> cropAndOrFlip(int ulx, int uly, int width, int height, List<TensorPair> data, double flipProbability) {
        Random random = new Random();
        return data.stream()
                   .map(tp->augment(tp,ulx,uly,width,height,random))
                   .collect(Collectors.toList());
    }
    
    public static TensorPair augment(TensorPair tp, int ulx, int uly, int width, int height, Random random) {
        
            Tensor t = tp.model_input;
            TensorShape newShape = new TensorShape(width,height,t.getDimension(TensorShape.Dimension.DEPTH));
            INDArray dataView = t.getValues().get(all(),all(),interval(ulx,ulx+width),interval(uly,uly+height));
            
            // copy data, a bit waste ful but needed
            dataView = dataView.dup(dataView.ordering());
            
            // left-right flip image by given probability
            if (random.nextDouble()<0.5) Nd4jUtil.flipRows(dataView);
            
            // construct tensor
            Tensor newTensor = new Tensor(dataView, newShape);
            return new TensorPair(newTensor,tp.model_output);
    }
}