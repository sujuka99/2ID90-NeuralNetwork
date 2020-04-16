package nl.tue.s2id90.dl.javafx;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.DEPTH;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.HEIGHT;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.WIDTH;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.function.BiFunction;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Images
 * Class able to generate BufferedImage from a Tensor
 * - image with values shown as text number
 * - image with values as a coloured pixel showing the relative
 *   value where white is highest value in tensor and black lowest
 * 
 * @author Roel van Engelen
 */
public class Images{
    // image settings
    private static final int SIZE=3;    // activation width & height
    
    /**
     * Convert 3D Tensor data to an image depicting relative neuron activation
     * 
     * Highest activation value has white colour
     * Lowest activation  value has black colour
     * 
     * @param data      Tensor 3D to be converted to image
     * @return          BufferedImage depicting Tensor
     */
    public static BufferedImage image_from_tensor_3d( Tensor data ){
        // calculate single image layer width and height
        int width  = data.getDimension( HEIGHT ) * SIZE;
        int height = data.getDimension( WIDTH  ) * SIZE;
        
        // create grayscale BufferedImage
        BufferedImage image = new BufferedImage( width, height, BufferedImage.TYPE_INT_RGB );
        
        // normalize Tensor values to be in range 0.0 - 1.0
        INDArray values = normalize_tensor( data.getValues().getRow(0) );
        
        if (data.getDimension(DEPTH)==3) {   
            fillColorImage(image.createGraphics(), data, values);
        } else {
            fillGrayImage(image.createGraphics(), data, values);
        }
        return image;
    }
    
    public static Image getFXImage(BufferedImage bi) {
        WritableImage wi = new WritableImage(bi.getWidth(), bi.getHeight());
        return SwingFXUtils.toFXImage(bi, wi);
    }
    
    private static void fillColorImage(Graphics2D g2, Tensor data, INDArray values) {
        fillImage(g2, data, (x,y) -> {
                float col0 = clamp(values.getFloat( new int[]{ 0, x, y } ));
                float col1 = clamp(values.getFloat( new int[]{ 1, x, y } ));
                float col2 = clamp(values.getFloat( new int[]{ 2, x, y } ));
                return new Color( col0, col1, col2 );
            }
        );
    }
    
    private static void fillGrayImage(Graphics2D g2, Tensor data, INDArray values) {
        fillImage(g2, data, (x,y) -> {
            float gray = clamp(values.getFloat( new int[]{ 0, x, y } ));
            return new Color( gray, gray, gray );
        });
    }
       
    private static void fillImage(Graphics2D g2, Tensor data, BiFunction<Integer,Integer,Color> getColor) {
        // loop over image width
        for( int x = 0 ; x < data.getDimension( HEIGHT ) ; x++ ){

            // loop over image height
            for( int y = 0 ; y < data.getDimension( WIDTH ) ; y++ ){
                
                g2.setColor(getColor.apply(x, y));
                
                // draw square activation pixel
                g2.fill(new Rectangle2D.Double( y * SIZE, x * SIZE, SIZE, SIZE));
            }
        }
    }
    
    /**
     * normalise INDArray that all values are in range 0.0 - 1.0
     * 
     * @param tensor INDArray with values to be normalised
     * @return       normalised INDArray
     */
    private static INDArray normalize_tensor( INDArray tensor ){
        
        // get lowest value in tensor
        float value_min = Nd4j.min( tensor ).getFloat( 0 );
        
        // if lowest value < 0.0 add value to whole tensor
        if( value_min < 0 ){
            
            // add -value_min to tensor -> lowest value will be 0.0
            tensor = tensor.add( -value_min );
        }
        
        // get max value in tensor
        float value_max = Nd4j.max( tensor ).getFloat( 0 );  
        
        // divide all values by value_max -> highest value will be 1.0
        tensor = tensor.div( value_max );
        
        return tensor;
    }
    
    /** @return clamped value to interval [0,1]. */
    private static float clamp(float x) { return Math.min(1, Math.max(0,x)); }
}