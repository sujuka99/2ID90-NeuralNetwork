package nl.tue.s2id90.dl.NN.layer;

import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.DEPTH;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.HEIGHT;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.WIDTH;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author Roel van Engelen
 */
public class Flatten extends Layer{
    
    private final int outputs;
    
    private final int depth;
    private final int width;
    private final int height;
    
    /**
     * Converts a xD tensor to a 1D tensor
     * @param name
     * @param input_shape 
     */
    public Flatten( String name, TensorShape input_shape ){
        super(name, input_shape, new TensorShape( input_shape.getNeuronCount() ) );
        
        this.outputs = input_shape.getNeuronCount();
        this.width  = input_shape.getShape( WIDTH  );
        this.height = input_shape.getShape( HEIGHT );
        this.depth  = input_shape.getShape( DEPTH  );
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
     * @param input
     * @return 
     */
    @Override
    public Tensor inference( Tensor input ){
        
        INDArray output = input.getValues().reshape( new long[] { input.getValues().shape()[ 0 ], outputs } );
        return new Tensor( output, outputShape );
    }

    /**
     * Calculate back-propagation
     * @param input
     * @return 
     */
    @Override
    public INDArray backpropagation( INDArray input ) {
        
        long batch_size = input.shape()[ 0 ];
        
        return input.reshape( new long[] { batch_size, depth, width, height } );
    }
}
