package nl.tue.s2id90.dl.input;

import java.io.IOException;
import java.util.Collections;
import java.util.List;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.input.CSVReader;
import nl.tue.s2id90.dl.input.InputReader;
import static java.lang.String.format;
import java.util.Map;

/**
 *
 * @author huub
 */
public class WineReader extends InputReader{
    
    /** simple enum to distinguish between the red wine and white wine dataset. */
    public enum Type { RED_WINE, WHITE_WINE };
    private final CSVReader csvReader;
    
     /**
     * Read all wine records from file and divide them over training (80%) 
     * and validation (20%) set.
     * 
     * @param batch_size amount of training data pairs in one batch
     * @param color  indicates whether to use white or red wine dataset
     * @throws java.io.IOException 
     */
    public WineReader( int batch_size, Type color) throws IOException{
        super( batch_size );
                        
        // MNIST training label and image file locations
        final String PREFIX = "data/wine/winequality";
        String fileName = format("%s-%s.csv", PREFIX, color==Type.RED_WINE?"red":"white");
        
        // read training and validation data
        csvReader = new CSVReader(';', fileName);
        List<TensorPair> data   = csvReader.getData();
        Collections.shuffle(data);
        
        // split dataset in 80% training, 20% validation
        int count = data.size(), split = (4*count)/5;
        setTrainingData(data.subList(0, split));
        setValidationData(data.subList(split,count));
    }
    
    public static void main(String[] args) throws IOException {
        WineReader wr = new WineReader(10, Type.RED_WINE);
    }
    
    @Override public Map<String,Object> getInfoMap() {
        Map map = super.getInfoMap();
        map.put("headers",csvReader.getHeaders());
        return map;
    }
}
