package nl.tue.s2id90.dl.input;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Read_MNIST_Images
 * read mnist data from file and create training and validation tensors
 * 
 * @author Roel van Engelen
 * @author Huub van de Wetering
 */
public class MNISTReader extends InputReader{

    final static private int IDX3 = 2051;
    final static private int IDX1 = 2049;
    
    /** returns a reader for primitives-MNIST dataset. 
     * @param batch_size amount of training data pairs in one batch
     * @return MNISTReader for the primitives dataset.
     * @throws java.io.IOException
     */
    public static MNISTReader primitives(int batch_size) throws IOException {
        return new MNISTReader("data/primitives",batch_size, 3);
    }
    
    /** returns a reader for fashion-MNIST dataset. 
     * @param batch_size amount of training data pairs in one batch
     * @return MNISTReader for the fashion dataset.
     * @throws java.io.IOException
     */
    public static MNISTReader fashion(int batch_size) throws IOException {
        return new MNISTReader("data/fashion",batch_size, 10);
    }

    /** returns a reader for the original MNIST dataset, containing images of handwritten digits. 
     * @param batch_size amount of training data pairs in one batch
     * @return MNISTReader for the original digits MNIST dataset.
     * @throws java.io.IOException
     */
    public static MNISTReader MNIST(int batch_size) throws IOException {
        return new MNISTReader("data/mnist",batch_size, 10);
    }
    
    /** returns a reader for the EMNIST letter dataset, containing images of handwritten letters. 
     * @param batch_size amount of training data pairs in one batch
     * @return MNISTReader for the original digits MNIST dataset.
     * @throws java.io.IOException
     */
    public static MNISTReader EMNISTLetters(int batch_size) throws IOException {
        return new MNISTReader("data/emnist/letters",batch_size, 26);
    }
    
    /* number of classes used in the classification */
    private final int numberOfClasses;
    
    /**
     * Reads all MINST image databases in the given folder.
     * 
     * @param folder   folder containing MNIST like databases.
     * @param batch_size amount of training data pairs in one batch
     * @param classes    amount of classes this model has
     * @throws java.io.IOException 
     */
    public MNISTReader( String folder, int batch_size, int classes ) throws IOException{
        super( batch_size );
        this.numberOfClasses = classes; 
                        
        // MNIST training label and image file locations
        String training_labels_file = folder+"/train-labels-idx1-ubyte.gz";
        String training_images_file = folder+"/train-images-idx3-ubyte.gz";
        
        // MNIST validation label and image file locations
        String validation_labels_file = folder+"/t10k-labels-idx1-ubyte.gz";
        String validation_images_file = folder+"/t10k-images-idx3-ubyte.gz";
        
        // read training and validation data
        setTrainingData(read_image_label_pairs( classes, training_images_file, training_labels_file ));
        setValidationData(read_image_label_pairs( classes, validation_images_file, validation_labels_file ));
    }
    
    /**
     * @return  number of classes used in the classification of this MNIST dataset.
     */
    public int getNumberOfClasses() {
        return numberOfClasses;
    }

    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private
    
    /**
     * Read all images and labels from file and create a list with all
     * tensor pairs
     * 
     * @param file_image image file location
     * @param file_label label file location
     * @return list with tensor pairs
     */
    private List<TensorPair> read_image_label_pairs( int classes, String file_image, String file_label ) throws IOException{
                
        // open label file
        InputStream stream_labels = get_Gzipped_inputstream( file_label );
        // open image file
        InputStream stream_images = get_Gzipped_inputstream( file_image );
                
        // validate labels stream to be IDX1
        if ( read_int( stream_labels ) != IDX1 ){
            
            throw new IllegalArgumentException("not an IDX1 file");
        }
        
        // validate images stream to be IDX3
        if ( read_int( stream_images ) != IDX3 ){
            
            throw new IllegalArgumentException("not an IDX3 file");
        }
        
        // read all images and labels from streams
        List<TensorPair> data = read_IDX1_IDX3_pair( classes, stream_labels, stream_images ); 
        
        // close both streams
        stream_labels.close();
        stream_images.close();
        
        return data;       
    }
    
    /**
     * read all images and labels from file and generate
     * list with tensor pairs
     * 
     * @param images IDX3 image InputStream
     * @param labels IDX1 label InputStream
     * @return list with all image & label tensor pairs
     */
    private List<TensorPair> read_IDX1_IDX3_pair( int classes, InputStream stream_labels, InputStream stream_images ) throws IOException{
        
        List<TensorPair> data = new ArrayList<>();
        
        // read labels
        int count_labels = read_int( stream_labels );
        byte[] labels    = read_bytes( stream_labels, count_labels );        
        // Tensor_Shape and INDArray shape for label
        TensorShape shape_label = new TensorShape( classes );
        int[]        indar_label = new int[]{ classes };
                 
        // read images
        int count_images = read_int( stream_images );
        int count_rows   = read_int( stream_images );
        int count_cols   = read_int( stream_images );        
        // Tensor_Shape and INDArray shape for image
        TensorShape shape_image = new TensorShape( count_rows, count_cols, 1 );
        int[]        indar_image = new int[]{ 1, 1, count_rows, count_cols };
        
        // validate images and labels count
        if ( count_images != count_labels ){
            
            throw new IllegalArgumentException( "Labels en Images count not equal" );
        }
        
        // read images, create label and image tensor and 
        // use them to create a training pair
        for( int x = 0 ; x < count_images ; x++ ){
            
            // create label tensor
            float[] label = get_empty_float_array( classes );
            
              // fix for datasets that have labels starting at 1, instead of 0.
            int fixedLabel = labels[x]==label.length? 0: labels[x];
            label[ fixedLabel ] = 1;
            
            Tensor tensor_label = new Tensor( Nd4j.create( label, indar_label, 'c' ), shape_label );
                        
            // create image tensor
            float[] image = read_image( stream_images, count_rows * count_cols );
            Tensor tensor_image = new Tensor( Nd4j.create( image, indar_image, 'c' ), shape_image );
            
            // add tensor training pair
            data.add(new TensorPair( tensor_image, tensor_label ) );
        }
                
        return data;        
    }
    
    /**
     * Author Huub van de Wetering
     *        Roel van Engelen
     * 
     * @return InputStream from a gzipped file. *
     */
    private InputStream get_Gzipped_inputstream( String file ) throws IOException{
        
        // try to open inputstream
        BufferedInputStream is = new BufferedInputStream( new FileInputStream( file ) );

        return new GZIPInputStream(is);
    }
        
    /**
     * Author Huub van de Wetering
     * 
     * @param is InputStream to read from
     * @return next integer on the input stream is
     * @throws IOException 
     */
    private int read_int( InputStream is ) throws IOException {
        
        // read next 4 bytes and convert to int
        return ByteBuffer.wrap( read_bytes( is, 4 ) ).getInt();
    }
        
    /**
     * Author Huub van de Wetering
     * 
     * reads noBytes from the input stream. *
     * 
     * @param is InputStream to read from
     * @param noBytes amount of bytes to read from stream
     * @return byte[] with bytes read
     * @throws IOException 
     */
    private byte[] read_bytes( InputStream is, int noBytes ) throws IOException{
        
        // create read buffer
        byte[] buffer = new byte[ noBytes ];
        int read = 0;
        
        // read bytes
        while( read < noBytes ){
            
            read += is.read( buffer, read, noBytes - read );
        }
        
        return buffer;
    }
    
    /**
     * Read size bytes from stream and create float[] as an image
     * 
     * @param is   InputStream to read from
     * @param size amount of bytes to read from stream
     * @return float[] with bytes read
     * @throws IOException 
     */
    private float[] read_image( InputStream is, int size ) throws IOException{
        
        // create float[] to store image data
        float[] ret = new float[ size ];
        // read size bytes from stream
        byte[]  data = read_bytes( is, size );
        
        // loop over read data and convert to float
        for( int x = 0 ; x < data.length ; x++ ){
            
            // data is read as a byte but is stored as an unsigned byte
            // convert byte to correct unsigned byte value
            ret[ x ] = ( ( data[ x ] & 0xff ) / 255f );     // -huub: replace 256f by 255f
        }
        
        return ret;
    }
    
    /**
     * create float[] with lenght size initialized to 0.0
     * 
     * @param size float[] size
     * @return return float[] with length size initialized to zero
     */
    private float[] get_empty_float_array( int size ){
        
        // create float[] of correct size
        float[] ret = new float[ size ];
        
        // initialize array to zero
        for( int x = 0 ; x< size ; x++ ){
            
            ret[ x ] = 0;
        }
        
        return ret;
    }
}
