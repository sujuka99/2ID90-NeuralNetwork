package nl.tue.s2id90.dl.NN.layer;

import static java.lang.String.format;
import java.util.Map;
import java.util.function.Supplier;
import nl.tue.s2id90.dl.NN.activation.Activation;
import nl.tue.s2id90.dl.NN.initializer.Initializer;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.DEPTH;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.HEIGHT;
import static nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension.WIDTH;
import nl.tue.s2id90.dl.json.JSONUtil;
import org.json.simple.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author Roel van Engelen
 */
public class Convolution2D extends Layer{
    
    private INDArray         kernel_weights;
    private INDArray         kernel_weights_flatten;
    private INDArray         kernel_bias;
    private INDArray         updated_kernel_weights;
    private INDArray         updated_kernel_bias;
        
    private final Activation activation;
    private INDArray         last_input;
    private INDArray         last_preoutput;
    private INDArray         im2col2d;
        
    private final int        kernels;
    private final int        kernel_size;
    private final int        padding;
    private final int        width;
    private final int        height;
    private final int        depth;
    
    /**
     * 
     * @param name
     * @param activation
     * @param input_shape
     * @param kernel_size
     * @param kernels
     */
    public Convolution2D( String name, TensorShape input_shape, int kernel_size, int kernels, Activation activation){
        super(name, input_shape, new TensorShape( input_shape.getShape( WIDTH ), 
                                                    input_shape.getShape( HEIGHT ), 
                                                    kernels ) );
        
        // TODO validate kernel size
        
        this.activation  = activation;
        this.kernels     = kernels;
        this.kernel_size = kernel_size;
        this.padding     = ( kernel_size - 1 ) / 2;
        this.width       = input_shape.getShape( WIDTH  );
        this.height      = input_shape.getShape( HEIGHT );
        this.depth       = input_shape.getShape( DEPTH  );
        
        // test whether output size is input size, here stride=1
        // outputSize = (inputSize+2*padding -kernelsize)/stride + 1
        if ( 2*padding - kernel_size +1 !=0) {
            throw new IllegalArgumentException(
               format("Convolution layer \"%s\" is initialized with an inappropriate kernel size (%d).",name,kernel_size)
            );
        }
    }
        
    ////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////// public
    
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
                
        // initialize bias and weights INDArrays
        // init bias with value zero
        kernel_bias    = Nd4j.zeros( new int[] { 1, kernels } );
        // init weights with initializer, keep a flattend weights and view with correct shape
        // initialize with 'f' ordering because of ND4j.gemm in backpropagation
        int fanIn = kernel_size * kernel_size * depth;
        int fanOut= kernel_size * kernel_size * kernels;
        kernel_weights_flatten = initializer.get_weight(fanIn, fanOut, new int[] { kernels, depth, kernel_size, kernel_size } );
        kernel_weights = kernel_weights_flatten.reshape( 'f', new int[] { kernels, depth, kernel_size, kernel_size } );
        
        // initialize bias and weight gradient accummulation INDArrays
        updated_kernel_bias    = kernel_bias.dup();
        updated_kernel_weights = kernel_weights.dup('c').assign(0);
    }
    
    /**
     * Calculate inference of input tensor
     * @param input
     * @return 
     */
    @Override
    public Tensor inference( Tensor input ){
                
        INDArray in = input.getValues();//.reshape( new int[] { 1, depth, width, height } );
        last_input  = in.dup();
        
        int batch_size = (int)in.shape()[ 0 ];
        
        INDArray col = Nd4j.createUninitialized(new int[] { batch_size, width, height, depth, kernel_size, kernel_size }, 'c');
        INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);
        
        Convolution.im2col( in, kernel_size, kernel_size, 1, 1, 1, 1, true, col2 );

        im2col2d = Shape.newShapeNoCopy(col, new int[] { batch_size * width * height, depth * kernel_size * kernel_size }, false);
        
        INDArray permutedW = kernel_weights.permute(3, 2, 1, 0);
        INDArray reshapedW = permutedW.reshape('f', kernel_size * kernel_size * depth, kernels );
        
        INDArray z = im2col2d.mmul( reshapedW );
        z = Shape.newShapeNoCopy(z, new int[] { width, height, batch_size, kernels }, true);
        z = z.permute( 2, 3, 1, 0 );
        
        z.addiRowVector( kernel_bias );
        
        last_preoutput = z.dup();        
        activation.activation( z );
        
        return new Tensor( z, outputShape );
    }

    /**
     * Calculate back-propagation
     * @param epsilon
     * @return 
     */
    @Override
    public INDArray backpropagation( INDArray epsilon ){
    
        int batch_size = (int)epsilon.shape()[ 0 ];
        INDArray weightGradView2df = Shape.newShapeNoCopy(updated_kernel_weights, new int[] {kernels, depth * kernel_size * kernel_size }, false).transpose();
        
        INDArray delta = activation.get_IActivation().backprop( last_preoutput, epsilon ).getFirst();
        
        delta = delta.permute(1, 0, 2, 3); //To shape: [outDepth,miniBatch,outH,outW]

        //Note: due to the permute in preOut, and the fact that we essentially do a preOut.muli(epsilon), this reshape
        // should be zero-copy; only possible exception being sometimes with the "identity" activation case
        INDArray delta2d = delta.reshape('c', new int[] { kernels, batch_size * height * width }); //Shape.newShapeNoCopy(delta,new int[]{outDepth,miniBatch*outH*outW},false);

        //Calculate weight gradients, using cc->c mmul.
        //weightGradView2df is f order, but this is because it's transposed from c order
        //Here, we are using the fact that AB = (B^T A^T)^T; output here (post transpose) is in c order, not usual f order
        Nd4j.gemm(im2col2d, delta2d, weightGradView2df, true, true, 1.0, 0.0);

        //Flatten 4d weights to 2d... this again is a zero-copy op (unless weights are not originally in c order for some reason)
        INDArray wPermuted = kernel_weights.permute(3, 2, 1, 0); //Start with c order weights, switch order to f order
        INDArray w2d = wPermuted.reshape('f', depth * kernel_size * kernel_size, kernels );

        //Calculate epsilons for layer below, in 2d format (note: this is in 'image patch' format before col2im reduction)
        //Note: cc -> f mmul here, then reshape to 6d in f order
        INDArray epsNext2d = w2d.mmul(delta2d); //TODO can we reuse im2col array instead of allocating new result array?
        INDArray eps6d = Shape.newShapeNoCopy(epsNext2d, new int[] { kernel_size, kernel_size, depth, width, height, batch_size }, true);

        //Calculate epsilonNext by doing im2col reduction.
        //Current col2im implementation expects input with order: [miniBatch,depth,kH,kW,outH,outW]
        //currently have [kH,kW,inDepth,outW,outH,miniBatch] -> permute first
        eps6d = eps6d.permute(5, 2, 1, 0, 4, 3);
        INDArray epsNextOrig = Nd4j.create(new int[] { depth, batch_size, height, width }, 'c');
        
        //Note: we are execute col2im in a way that the output array should be used in a stride 1 muli in the layer below... (same strides as zs/activations)
        INDArray epsNext = epsNextOrig.permute(1, 0, 2, 3);
//        Convolution.col2im(eps6d, epsNext, 1, 1, 1, 1, height, width );    // original
//        if (true) throw new UnsupportedOperationException("this code has to be ported to nd4j 1.000");
        Convolution.col2im(eps6d, epsNext, 1, 1, 1, 1, height, width, 1, 1); // fjust a guess!!

        delta2d.sum( updated_kernel_bias, 1); //biasGradView is initialized/zeroed first in sum op
        
        return epsNext;
    }
    
//    /**
//     * Update bias and weights
//     * 
//     * @param learning_rate
//     * @param batch_size
//     */
//    @Override
//    public void updateLayer( float learning_rate, int batch_size ){
//        
//        float learning = learning_rate;
//        // TODO learning rate should be divided by batch_size but
//        // learning rate gets to low compared to fully connected layers
//        // does convolution already average over batch???
//        
//        // update bias with accumulated gradients
//        updated_kernel_bias.muli( learning );
//        
//        // update bias
//        Nd4j.getBlasWrapper().level1().axpy( kernel_bias.length(), -1.0, updated_kernel_bias, kernel_bias );
//        // re initialize update_bias array
//        updated_kernel_bias = Nd4j.zeros( new int[] { 1, kernels } );
//        
//        // update weights with accumulated gradients
//        updated_kernel_weights.muli( learning ); 
//        // toDo learning rate gets to small when dividing by batch size??? 
//        Nd4j.getBlasWrapper().level1().axpy( kernel_weights.length(), -1.0, updated_kernel_weights, kernel_weights );
//        
//        // re initialize update_weights array
//        updated_kernel_weights = kernel_weights.dup().assign(0);
//    }
    
    private UpdateFunction biasUpdater, weightsUpdater;
    @Override
    public void updateLayer(Supplier<UpdateFunction> createUpdateFunction, double learning_rate, int batch_size) {
        if (biasUpdater==null) biasUpdater=createUpdateFunction.get();
        if (weightsUpdater==null) weightsUpdater=createUpdateFunction.get();
        
         // duplicating the gradients updated_bias, since update() method might change them otherwise
        INDArray biasGrad = updated_kernel_bias.dup(updated_kernel_bias.ordering());
        biasUpdater.update(kernel_bias, true, learning_rate, batch_size, biasGrad);
//        updated_kernel_bias.assign(0);
        
        INDArray weightsGrad = updated_kernel_weights.dup(updated_kernel_weights.ordering());
        weightsUpdater.update(kernel_weights, false, learning_rate, batch_size, weightsGrad);
//        updated_kernel_weights.assign(0);
    }
    
    //<editor-fold defaultstate="collapsed" desc="serialization">
    @Override
    public JSONObject json() {
        JSONObject jo = super.json();
        jo.put("activation", activation.json());
        jo.put("kernel_size", kernel_size);
        jo.put("kernels",kernels);
        jo.put("weights",JSONUtil.toJson(kernel_weights_flatten));
        jo.put("bias",JSONUtil.toJson(kernel_bias));
        return jo;
    }
    
    /** non-public, package method to set weights directly, used during
     * de-serialization.
     */
    void setWeights(float[] w) {
        int expectedSize =kernels*depth*kernel_size*kernel_size;
        if (w.length!= expectedSize) {
            throw new IllegalArgumentException(format("Weights array has length %d, but expected %d!", w.length, expectedSize));
        }
        //        kernel_weights_flatten = new NDArray(w);
        kernel_weights_flatten = Nd4j.create(w);
        kernel_weights = kernel_weights_flatten.reshape( 'f', new int[] { kernels, depth, kernel_size, kernel_size } );
    }
    
    /** @return weights as a flat nd array. Returns a copy. */
    public INDArray getWeightsAsNDArray() {
        return kernel_weights_flatten.dup();
    }
    
    /** non-public, package method to set biases directly, used during
     * de-serialization.
     */
    void setBias(float[] w) {
        if (w.length!= kernels) {
            throw new IllegalArgumentException(format("Bias array has length %d, but expected %d!",w.length,kernels));
        }
        //        kernel_bias = new NDArray(w);
        kernel_bias = Nd4j.create(w);
    }
//</editor-fold>
    
    @Override
    public Map<String, Object> getInfoMap() {
        Map result= super.getInfoMap();
        result.put("activation", activation.getClass().getSimpleName());
        result.put("kernel size",kernel_size);
        result.put("kernels", kernels);
        return result;
    }
    
       
    //<editor-fold defaultstate="collapsed" desc="auxiliary functions for GradientChecker">
     @Override
     INDArray getGradient() {
        INDArray flatWeights = updated_kernel_weights.reshape( 'f', new int[] { 1, kernels* depth* kernel_size* kernel_size } );
        return Nd4j.concat(1,flatWeights,updated_kernel_bias);
    }

    @Override
     Double getParameter(int i) {
        return Layers.getParameter(i, kernel_weights_flatten,kernel_bias);
    }
    
    @Override
     boolean setParameter(int i, double d) {
        return Layers.setParameter(i, d, kernel_weights_flatten,kernel_bias);
    }
    
    @Override
     int getNumberOfParameters() {
        return Layers.getNumberOfParameters(kernel_weights_flatten, kernel_bias);
    }
    
    @Override
     void clearParameters() {
        Layers.clearParameters(kernel_weights_flatten,kernel_bias);
    }
    //</editor-fold>
}
