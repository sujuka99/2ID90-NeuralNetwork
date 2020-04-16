package nl.tue.s2id90.dl.NN.tensor;

import nl.tue.s2id90.dl.json.JSONable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import nl.tue.s2id90.dl.json.JSONUtil;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

/**
 * Tensor_Shape
 * class containing a tensor shape data able
 * to verify data shape equality
 * 
 * @author Roel van Engelen
 */
public class TensorShape implements JSONable {
        
    // holds shape dimensions
    private final int[] shape;
    
    /**
     * 3D tensor shape
     * 
     * @param width
     * @param height
     * @param depth 
     */
    public TensorShape( int width, int height, int depth ){
        
        shape = new int[ 4 ];
        shape[ Dimension.BATCH.getPosition()  ] = 1;
        shape[ Dimension.WIDTH.getPosition()  ] = width;
        shape[ Dimension.HEIGHT.getPosition() ] = height;
        shape[ Dimension.DEPTH.getPosition()  ] = depth;
    }
        
    /**
     * 1D tensor shape
     * 
     * @param size 
     */
    public TensorShape( int size ){
        
        shape = new int[ 2 ];
        shape[ Dimension.BATCH.getPosition() ] = 1;
        shape[ Dimension.SIZE.getPosition()  ] = size;
    }
    
    /**
     * Get requested tensor dimension: BATCH, SIZE, WIDTH, HEIGHT, DEPTH
     * 
     * @param dimension requested dimension
     * @return requested dimension value
     */
    public int getShape( Dimension dimension ){
        
        return shape[ dimension.getPosition() ];
    }
    
    /**
     * get shape dimensions array
     * 
     * @return int array describing tensor shape
     */
    public int[] getShape(){
        
        return shape;
    }
        
    /**
     * Validate that this shape is the same as shape
     * 
     * @param shape int[] shape to compare this shape with
     * @return true if shapes are equal, false if not
     */
    public boolean isCorrectShape( int[] shape ){
        
        // verify shape #dimensions are the same
        if( this.shape.length != shape.length ){
            
            // shapes not equal
            return false;
        }
        
        // verify all dimensions are equal
        // first dimension is batch and might be different
        for( int x = 1 ; x < this.shape.length ; x++ ){
            
            if( this.shape[ x ] != shape[ x ] ){
                
                // shapes not equal
                return false;
            }
        }
        
        // shapes are the same
        return true;
    }
    
    /**
     * Validate that this shape is the same as shape
     * 
     * @param shape long[] shape to compare this shape with
     * @return true if shapes are equal, false if not
     */
    public boolean isCorrectShape( long[] shape ){
        
        // verify shape #dimensions are the same
        if( this.shape.length != shape.length ){
            
            // shapes not equal
            return false;
        }
        
        // verify all dimensions are equal
        // first dimension is batch and might be different
        for( int x = 1 ; x < this.shape.length ; x++ ){
            
            if( this.shape[ x ] != shape[ x ] ){
                
                // shapes not equal
                return false;
            }
        }
        
        // shapes are the same
        return true;
    }
    
    /**
     * is this Tensor 1D or 3D
     * 
     * @return true if 3D else false
     */
    public boolean is3D(){
        
        return shape.length > 2;
    }
    
    /**
     * Get total shape count
     * 
     * @return 
     */
    public int getNeuronCount(){
        
        int count = 1;
        for( int x = 1 ; x < shape.length ; x++ ){
            
            count *= shape[ x ];
        }
        
        return count;
    }
    
    /**
     * Format shape as (x,x,x)
     * 
     * @return formatted shape
     */
    public String shapeToString(){           // Huub: removed trailing comma
        return IntStream.of(shape).boxed()
                .map(i->i.toString())
                .collect(Collectors.joining(",", "(",")"));
    }
    
    @Override                                   // Huub:added this for more easily debugging shapes
    public String toString() {
        return shapeToString();
    }
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////// enum

    @Override
    public JSONObject json() {
        JSONObject jo = new JSONObject();
        jo.put("shape", Arrays.stream(shape).boxed().collect(Collectors.toList()));
        return jo;
    }
    
    /**
     * Enum describing all tensor shape dimensions
     * BATCH, SIZE, WIDTH, HEIGHT, DEPTH
     * 
     * used locations are the same as used by Nd4j and therefore able to
     * verify INDArray shape as well as Tensor_Shape shape
     */
    public enum Dimension{
        BATCH  ( 0 ),  // batch dimension
        SIZE   ( 1 ),  // 1d dimension
        WIDTH  ( 2 ),  // 3d width  dimension
        HEIGHT ( 3 ),  // 3d height dimension
        DEPTH  ( 1 );  // 3d depth  dimension
        
        private final int position;
        
        /**
         * create new enum with corresponding position
         * 
         * @param position 
         */
        private Dimension( int position ){
            
            this.position = position;
        }
        
        /**
         * get array position
         * 
         * @return get this dimension location in shape array
         */
        public int getPosition(){
            
            return position;
        }
    }
    
    public static TensorShape fromJson(JSONObject jo) {
        JSONArray array = (JSONArray)jo.get("shape");
        List<Long> shape = JSONUtil.toList(array, Long.class);
        if (shape.size()==2) {
            int size=shape.get(Dimension.SIZE.getPosition()).intValue();
            return new TensorShape(size);
        } else if (shape.size()==4) {
            int height=shape.get(Dimension.HEIGHT.getPosition()).intValue();
            int width=shape.get(Dimension.WIDTH.getPosition()).intValue();
            int depth=shape.get(Dimension.DEPTH.getPosition()).intValue();
            return new TensorShape(width, height, depth);
        }
       throw new IllegalStateException("Unknown tensor_shape: "+shape);
    }
}
