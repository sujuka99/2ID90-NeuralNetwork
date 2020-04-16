package nl.tue.s2id90.dl.NN.tensor;

import lombok.Getter;
import nl.tue.s2id90.dl.NN.tensor.TensorShape.Dimension;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Tensor
 * data class holding INDArray with data and data shape
 * 
 * @author Roel van Engelen
 */
public class Tensor{
    
    @Getter private final INDArray     values;
    @Getter private final TensorShape shape;
           
    /**
     * Create new Tensor width INDArray data and Tensor_Shape shape
     * 
     * @param data INDArray containing tensor data
     * @param shape Tensor_Shape with tensor shape
     */
    public Tensor( INDArray data, TensorShape shape ){
        
        this.values = data;        
        this.shape  = shape;
    }
    
    /**
     * Get size of Dimension
     * 
     * @param dimension requested dimension: BATCH, WIDTH, HEIGHT, SIZE, DEPTH
     * @return dimension size
     */
    public int getDimension( Dimension dimension ){
        
        return shape.getShape( dimension );
    }
    
    /**
     * return a copy of this tensor
     * 
     * @return 
     */
    public Tensor getCopy(){
        
        return new Tensor( values.dup(), shape );
    }
    
    /**
     * Validates this Tensor input shape is equal to shape
     * and INDArray has the same shape as shape
     * 
     * @param shape Tensor_Shape to compare shape with
     * @return true is shapes are equal, false if not
     */
    public boolean isCorrectShape( TensorShape shape ){
        
        return shape.isCorrectShape( this.shape.getShape() ) && 
               shape.isCorrectShape(values.shape() );
    }
    
    /**
     * is this Tensor 1D or 3D
     * 
     * @return true if 3D else false
     */
    public boolean is3D(){
        
        return shape.is3D();
    }
    
    @Override
    public String toString() {
        return values.toString();
    }
}
