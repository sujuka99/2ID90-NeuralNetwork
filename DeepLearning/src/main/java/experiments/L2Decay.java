package experiments;

import java.util.function.Supplier;
import nl.tue.s2id90.dl.NN.optimizer.update.UpdateFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import static org.nd4j.linalg.ops.transforms.Transforms.pow;


public class L2Decay implements UpdateFunction {
    double decay;
    UpdateFunction f;
    
    //constuctor 
    public L2Decay(Supplier<UpdateFunction> supplier, double decay){
        this.decay = decay;
        this.f = supplier.get();
    }
    
    @Override 
        public void update(INDArray array, boolean isBias, double learningRate, int batchSize, INDArray gradient) {
            f.update(array, isBias, learningRate, batchSize, gradient);
            
            // Only apply L2Decay when no bias
            if (!isBias) {
                array.addi(pow(array, 2).mul(learningRate - decay));//we do step decay by reducing by a half
            }
        }
}
