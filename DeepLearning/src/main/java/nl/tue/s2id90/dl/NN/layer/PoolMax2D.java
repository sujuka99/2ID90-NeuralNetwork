package nl.tue.s2id90.dl.NN.layer;

import static java.lang.String.format;
import java.util.Map;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.DEPTH;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.HEIGHT;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.WIDTH;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
//import org.nd4j.linalg.api.ops.impl.transforms.convolution.Pooling2D;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

/**
 *
 * @author Roel van Engelen
 */
public class PoolMax2D extends Layer{
    
    private INDArray  last_input;
    
    private final int stride;
    
    private final int in_width;
    private final int in_height;
    private final int out_width;
    private final int out_height;
    private final int depth;
    
    /**
     * 
     * @param name
     * @param input_shape
     * @param stride 
     */
    public PoolMax2D( String name, TensorShape input_shape, int stride ){
        super(name, input_shape, 
               new TensorShape( input_shape.getShape( WIDTH  ) / stride, 
                                 input_shape.getShape( HEIGHT ) / stride, 
                                 input_shape.getShape( DEPTH  ) ) );  
        
        this.stride      = stride;
        this.in_width    = input_shape.getShape( WIDTH  );
        this.in_height   = input_shape.getShape( HEIGHT );
        this.out_width   = input_shape.getShape( WIDTH  ) / stride;
        this.out_height  = input_shape.getShape( HEIGHT ) / stride;
        this.depth       = input_shape.getShape( DEPTH  );   
        
        if( in_width!=stride*out_width || in_height!=stride*out_height){            
            throw new IllegalArgumentException(
                format("In layer \"%s\": stride(%d) has to be a divisor of both image width(%d) and height(%d)",
                        name, stride, in_width, in_height )
            );
        }
    }
        
    /**
     * should activation image show actual values or relative activations
     * 
     * @return true for actual values, false for relative activations
     */
    @Override
    public boolean showValues(){
        
        return false;
    }
    
    /**
     * Initialize bias and weights
     * @param initializer 
     */
    @Override
    public void initializeLayer( Initializer initializer ){
        
        // Nothing to initialize
    }
    
    /**
     * Calculate inference of input tensor
     * 
     * @param input
     * @return 
     */
    @Override
    public Tensor inference( Tensor input ){
                
        // add batch dimension input shape is [ depth, in_width, in_height ]
        // and has to be [ 1, depth, in_width, in_height ]
        INDArray in  = input.getValues();
        last_input   = in.dup();
        
        int batch_size = (int)in.shape()[ 0 ];
                
        // create uninitialized INDArray with 1 * out_height * out_width * depth neurons
        INDArray out = Nd4j.createUninitialized( batch_size * out_height * out_width * depth );
        out=out.reshape( new int[]{ batch_size, depth, out_width, out_height } ); // added for nd4j 1.0.0;
        
        
        // Apply max pool with #stride stride, #stride kernelsize and store result in INDArray out
//        if (true) throw new UnsupportedOperationException("this code has to be ported to nd4j 1.000");
        // the Next line has to be ported to nd4j 1.00
        Convolution.pooling2D(
                in,                                 // image
                stride, stride,                     // kh, kw     // kernel
                stride, stride,                     // sy, sx     // stride
                0, 0,                               // ph, pw     // padding
                1, 1,                               // dh, dw,    // dilation, added for nd4j 1.0.0
                true,                               // is sameMode
                Pooling2D.Pooling2DType.MAX,        // pooling type
                Pooling2D.Divisor.INCLUDE_PADDING,  // pooling divisor, added for nd4j 1.0.0
                0.0,                                // extra
                out_height,                         // virtual height
                out_width,                          // virtual width
                out );                              // output
        // NOTE! newer versions use new pooling2D ( dilation is added )
   
        // INDArray out has shape [ depth * out_width * out_height ]
        // and has to be [ 1, depth, out_width, out_height ]
        return new Tensor( out.reshape( new int[]{ batch_size, depth, out_width, out_height } ), outputShape );
    }

    /**
     * Calculate back-propagation
     * @param epsilon
     * @return 
     */
    @Override
    public INDArray backpropagation( INDArray epsilon ) {
        
        int batch_size = (int)epsilon.shape()[ 0 ];
        
        boolean cOrderStrides = false;
        if( epsilon.ordering() != 'c'){
            
            epsilon = epsilon.dup('c');
            cOrderStrides = true;
        }
        
        INDArray col6d;
        INDArray col6dPermuted;
        INDArray epsilon1d;
        if (cOrderStrides) {
            //"Dense/Output layer above strides... i.e., standard c-order strides
            col6d = Nd4j.create(new int[] {batch_size, depth, out_height, out_width, stride, stride }, 'c');
            col6dPermuted = col6d.permute(0, 1, 4, 5, 2, 3);
            epsilon1d = epsilon.reshape('c', ArrayUtil.prod( epsilon.length()), 1); //zero copy reshape
        } else {
            //"CNN layer above" strides...
            col6d = Nd4j.create(new int[] { depth, batch_size, out_height, out_width, stride, stride }, 'c');
            col6dPermuted = col6d.permute(1, 0, 4, 5, 2, 3);

            INDArray epsilonTemp = epsilon.permute(1, 0, 2, 3);
            epsilon1d = epsilonTemp.reshape('c', new int[] {ArrayUtil.prod(epsilon.length()), 1}); //Should be a zero-copy reshape always
        }
        
        INDArray col2d = col6d.reshape('c', batch_size * depth * out_height * out_width, stride * stride);
                        
        //Execute im2col, then reshape to 2d. Note rows are in a different order for cOrderStrides true vs false cases
        Convolution.im2col( last_input, stride, stride, stride, stride, 0, 0, true, col6dPermuted );
        INDArray isMax = Nd4j.getExecutioner().execAndReturn( new IsMax( col2d, 1 ) );
        isMax.muliColumnVector(epsilon1d);
        
        
        //Finally: we want the output strides for the epsilons to match the strides in the activations from the layer below
        //Assuming the layer below is a CNN layer (very likely) we want [H*W, depth*H*W, W, 1] instead of the standard
        // c-order [depth*H*W, H*W, W, 1] strides
        //To achieve this: [depth, miniBatch, H, W] in c order, then permute to [miniBatch, depth, H, W]
        //This gives us proper strides of 1 on the muli...
        INDArray tempEpsilon = Nd4j.create(new int[] {depth, batch_size, in_height, in_width}, 'c');
        INDArray outEpsilon = tempEpsilon.permute(1, 0, 2, 3);
//        if (true) throw new UnsupportedOperationException("this code has to be ported to nd4j 1.000");
        // the Next line has to be ported to nd4j 1.00
//        Convolution.col2im(col6dPermuted, outEpsilon, stride, stride, 0, 0, in_height, in_width);    // original
        Convolution.col2im(col6dPermuted, outEpsilon, stride, stride, 0, 0, in_height, in_width, 1, 1 );  // converted to nd4j 1.0.0

        return outEpsilon;    
    }
    
    @Override
    public JSONObject json() {
        JSONObject jo = super.json();
        jo.put("stride", stride);
        return jo;
    }  
    
    @Override
    public Map<String, Object> getInfoMap() {
        Map result= super.getInfoMap();
        result.put("stride", stride);
        return result;
    }
}
