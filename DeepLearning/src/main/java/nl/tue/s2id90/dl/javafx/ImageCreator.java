package nl.tue.s2id90.dl.javafx;

import nl.tue.s2id90.dl.NN.tensor.Tensor;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.DEPTH;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.HEIGHT;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.WIDTH;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Image_Creator
 * Class able to generate BufferedImage from a Tensor
 * - image with values shown as text number
 * - image with values as a coloured pixel showing the relative
 *   value where white is highest value in tensor and black lowest
 * 
 * @author Roel van Engelen
 */
public class ImageCreator{
    
    // image settings
    private final int neuron_size;    // activation width & height
    private final int pixel_spacing;  // space between two activations
    private final int padding_border; // space between outer activation and 
                                      // image border
    
    // image colouts
    private final Color colour_text       = Color.BLACK;
    private final Color colour_background = Color.decode("#ffb6c1");  // light pink
    
    /**
     * Initialise new Image_Creator with given image settings
     * 
     * @param spacing     spacing between two pixels
     * @param border      spacing between outer pixel and border
     * @param neuron_size square size of neuron in relative activation
     */
    public ImageCreator( int spacing, int border, int neuron_size ){
        
        this.neuron_size    = neuron_size;        
        this.pixel_spacing  = spacing;        
        this.padding_border = border;
    }
    
    ////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////// public
    
    /**
     * Determine what kind of image has to be generated 1D / 3D and:
     * - image with values shown as text number
     * - image with values as a coloured pixel showing the relative
     *   value where white is highest value in tensor and black lowest
     * 
     * @param data        tensor with image data
     * @param show_values boolean show values if true, relative activation if false
     * @param max_width   max image width
     * @param image_row
     * @return            generated image
     */
    public BufferedImage create_image( Tensor data, boolean show_values, int max_width, int image_row ){
                      
        // determine tensor dimensionality
        if( data.is3D() ){
            
            // 3D tensor
            return image_from_tensor_3d( data, max_width, image_row );
        }
            
        // show numerical values instead of relative activations
        if( show_values ){
            
            return image_from_tensor_1d_show_values( data, max_width, image_row );
        }
        
        // 1D tensor
        return image_from_tensor_1d( data, max_width, image_row ); 
    }
    
    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private
    
    /**
     * Convert 1D Tensor data to an image with every activation as an value
     * 
     * @param data      Tensor 1D to be converted to image
     * @param max_width max image width
     * @return          BufferedImage depicting Tensor
     */
    private BufferedImage image_from_tensor_1d_show_values( Tensor data, int max_width, int image_row ){
        
        // activation pixels in image
        int pixels = data.getShape().getNeuronCount();
        int text_height = 15;
        
        // calculate image width and height
        int width = Math.min( 200, max_width );
        int height = ( padding_border * 3 ) + pixels * ( text_height + pixel_spacing );
        
        // create grayscale BufferedImage
        BufferedImage ret = get_empty_image( width, height );
        // Graphics2D able to draw on BufferedImage
        Graphics2D g2 = ret.createGraphics();
        
        g2.setColor( colour_text );
        // loop over all neuron values
        for( int i = 0 ; i < pixels ; i++ ){
            
            // get neuron activation value
            String value = ""+ data.getValues().getRow( image_row ).getFloat( i );
            // draw textual number value on image
            g2.drawString( value, padding_border, text_height + padding_border + ( i * ( text_height + pixel_spacing ) ) );            
        }
        return ret;
    }
    
    /**
     * Convert 1D Tensor data to an image depicting relative neuron activation
     * Highest activation value has white colour
     * Lowest activation  value has black colour
     * 
     * @param data      Tensor 1D to be converted to image
     * @param max_width max image width
     * @return          BufferedImage depicting Tensor
     */
    private BufferedImage image_from_tensor_1d( Tensor data, int max_width, int image_row ){
            
        // activation pixels in image
        int pixels = data.getShape().getNeuronCount();
        
        // calculate image width
        int width = 2 * padding_border + pixels * neuron_size + ( pixels - 1 ) * pixel_spacing;
        int height = 2 * padding_border;
        
        // check that width is not > max_width
        if( width > max_width ){
            
            width = max_width;
            
            // calculate amount of activation pixels per row
            int pixel_per_row = ( width - 2 * padding_border ) / ( neuron_size + pixel_spacing );
            // calculate image height
            height += ( ( pixels / pixel_per_row ) + 1 ) * ( neuron_size + pixel_spacing );
        }else{
            
            // image is only one activation pixel high
            height += neuron_size;
        }
        
        // create grayscale BufferedImage
        BufferedImage ret = get_empty_image( width, height );
        // Graphics2D able to draw on BufferedImage
        Graphics2D g2 = ret.createGraphics();
        
        // normalize Tensor values to be in range 0.0 - 1.0
        INDArray values = normalize_tensor( data.getValues().getRow( image_row ) );
        
        // keep track of activation pixel x&y location
        int x = padding_border;
        int y = padding_border;
        
        // loop over all neuron values
        for( int i = 0 ; i < pixels ; i++ ){
            
            // get neuron activation value
            float col = values.getFloat( i );
            // set colour
            g2.setColor( new Color( col, col, col ) );
            // draw square activation pixel
            g2.fill(new Rectangle2D.Double(x, y,
                               neuron_size,
                               neuron_size));
            
            // calculate next activation pixel location
            x += neuron_size + pixel_spacing;
            
            // check that next activation pixel is visible
            if( x + padding_border >= max_width ){
                
                // set activation pixel location on next row
                x  = padding_border;
                y += neuron_size + pixel_spacing;
            }
        }
        
        // return image
        return ret;
    }
    
    /**
     * Convert 3D Tensor data to an image depicting relative neuron activation
     * 
     * Highest activation value has white colour
     * Lowest activation  value has black colour
     * 
     * @param data      Tensor 3D to be converted to image
     * @param max_width max image width
     * @return          BufferedImage depicting Tensor
     */
    private BufferedImage image_from_tensor_3d( Tensor data, int max_width, int image_row ){
            
        int pixel_spacing_3d = 0;
        // calculate single image layer width and height
        int image_width  = data.getDimension( HEIGHT ) * ( neuron_size + pixel_spacing_3d ) + padding_border;
        int image_height = data.getDimension( WIDTH  ) * ( neuron_size + pixel_spacing_3d ) + padding_border;
        // calculate amount of layer images per row
        int images_per_row = max_width / image_width;
        
        // width and height is at leas padding_border
        int width  = padding_border;
        int height = padding_border;
        
        // determine if multiple rows with layer images are needed
        if( data.getDimension( DEPTH ) > images_per_row ){
            
            // multiple rows with layer images
            width  += images_per_row * image_width;
            height += ( data.getDimension( DEPTH ) / images_per_row + 1 ) * image_height;
        }else{
            
            // only one row with layer images
            width  += data.getDimension( DEPTH ) * image_width;
            height += image_height;
        }
        
        // create grayscale BufferedImage
        BufferedImage ret = get_empty_image( width, height );
        // Graphics2D able to draw on BufferedImage
        Graphics2D g2 = ret.createGraphics();
        
        // normalize Tensor values to be in range 0.0 - 1.0
        INDArray values = normalize_tensor( data.getValues().getRow( image_row ) );
        
        int loc_x = padding_border;
        int loc_y = padding_border;
        
        if (data.getDimension(DEPTH)==3) {   
            createColorImage(g2, data, values, loc_x, loc_y, pixel_spacing_3d);
        } else {
            createGrayImage(g2, data, values, loc_x, loc_y, pixel_spacing_3d, image_width, width, image_height);
        }
        
        // return image
        return ret;
    }

    private void createGrayImage(Graphics2D g2, Tensor data, INDArray values,
            int loc_x, int loc_y, int pixel_spacing_3d, int image_width, int width, int image_height)
    {
        // loop over all layers in tensor ( depth )
        for( int z = 0 ; z < data.getDimension( DEPTH ) ; z++ ){
            
            // loop over image width
            for( int x = 0 ; x < data.getDimension( HEIGHT ) ; x++ ){
                
                // loop over image height
                for( int y = 0 ; y < data.getDimension( WIDTH ) ; y++ ){
                    
                    // get neuron activation value
                    float col = values.getFloat( new int[]{ z, x, y } );
                    // set colour
                    g2.setColor( new Color( col, col, col ) );
                    // draw square activation pixel
                    g2.fill(new Rectangle2D.Double( loc_x + ( y * ( neuron_size + pixel_spacing_3d ) ),
                            loc_y + ( x * ( neuron_size + pixel_spacing_3d ) ),
                            neuron_size,
                            neuron_size));
                }
            }
            
            // update x start location of sinle layer image
            loc_x += image_width;
            // check that next layer image pixel is visible
            if( loc_x + padding_border >= width ){
                
                // next layer image is on next row
                loc_x = padding_border;
                loc_y += image_height;
            }
            
        }
    }
    
    private void createColorImage(Graphics2D g2, Tensor data, INDArray values, int loc_x, int loc_y, int pixel_spacing_3d) {
            // loop over image width
            for( int x = 0 ; x < data.getDimension( HEIGHT ) ; x++ ){
                
                // loop over image height
                for( int y = 0 ; y < data.getDimension( WIDTH ) ; y++ ){
                    
                    // get neuron activation value
                    float col0 = values.getFloat( new int[]{ 0, x, y } );
                    float col1 = values.getFloat( new int[]{ 1, x, y } );
                    float col2 = values.getFloat( new int[]{ 2, x, y } );
                    // set colour
                    try {
                        g2.setColor( new Color( col0, col1, col2 ) );
                    } catch(IllegalArgumentException e) {
                        System.err.println(String.format("Illegal color (%f,%f,%f)",col0,col1,col2));
                        g2.setColor(new Color(1,0,0)); // alarm red :-)
                    }
                    // draw square activation pixel
                    g2.fill(new Rectangle2D.Double( loc_x + ( y * ( neuron_size + pixel_spacing_3d ) ),
                            loc_y + ( x * ( neuron_size + pixel_spacing_3d ) ),
                            neuron_size,
                            neuron_size));
                }
            }
    }
    
    /**
     * normalise INDArray that all values are in range 0.0 - 1.0
     * 
     * @param tensor INDArray with values to be normalised
     * @return       normalised INDArray
     */
    private INDArray normalize_tensor( INDArray tensor ){
        
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
    
    /**
     * Create empty image with background colour
     * 
     * @param width  image width
     * @param height image height
     * @return       empty image
     */
    private BufferedImage get_empty_image( int width, int height ){
        
        // create grayscale BufferedImage
        BufferedImage ret = new BufferedImage( width, height, BufferedImage.TYPE_INT_RGB );
        // Graphics2D able to draw on BufferedImage
        Graphics2D g2 = ret.createGraphics();
        
        // draw background colour
        g2.setPaint ( colour_background );
        g2.fillRect ( 0, 0, ret.getWidth(), ret.getHeight() );
        
        return ret;
    }
}
