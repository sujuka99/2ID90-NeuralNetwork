package nl.tue.s2id90.dl.input;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import static java.lang.String.format;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import static java.util.stream.Collectors.toList;
import static java.util.stream.StreamSupport.stream;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.NN.tensor.TensorShape;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author huub
 */
public class CSVReader {
    private final  List<TensorPair> data;
    private List<String> headers;
    private final char DELIMITER;
    
    /**
     * Read all csv records from the files. The first line is interpreted as
     * a header containing the field names. All fields are read in as floats.
     * 
     * 
     * @param files   list of csv file names
     * @param delimiter delimiter used in files, e.g. ',' or ';'
     * @throws java.io.IOException 
     */
    public CSVReader(char delimiter, String ... files) throws IOException{
        this.DELIMITER = delimiter;
        data   = readData(Arrays.asList(files));
    }
    
    /** @return list of field names in order of appearance in the files. */
    public List<String> getHeaders() {
        return headers;
    }
    
    /** returns list of tensor pairs, one for each input record. The input tensor
     * contains all fields but the last one, the output tensor contains only the last field.
     * @return list of tensor pairs as read from the input csv files.
     */
    public List<TensorPair> getData() {
        return data;
    }

    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////// private 
    
    private List<TensorPair> readData(List<String> files) {
        return files.stream()
                .map(f->new File(f))
                .flatMap(batch->readData(batch).stream())
                .collect(toList());
    }

    private List<TensorPair> readData(File file) {
        try(Reader in = new FileReader(file)) {
            CSVParser parser= CSVFormat.RFC4180
                    .withDelimiter(DELIMITER)
                    .withFirstRecordAsHeader()
                    .parse(in);
            
            Iterable<CSVRecord> records = parser;
            headers = getHeaders(parser.getHeaderMap());
            return stream(records.spliterator(), false)
                    .map(record -> toTensorPair(record))
                    .collect(Collectors.toList());
        } catch (IOException e) {
                System.err.println(format("error reading %s\n%s", file, e));
                return null;
        }
    }
    
    private TensorPair toTensorPair(CSVRecord csv) {
        final int SIZE = csv.size();
        int[]        indar_label = new int[]{ 1 };
        int[]        indar_image = new int[]{ 1, SIZE-1 };

        float[] labelf = new float[]{getFloat(csv,SIZE-1)};
        TensorShape ts1 = new TensorShape(1);                
        Tensor t1 = new Tensor( Nd4j.create( labelf, indar_label, 'f' ), ts1 );  

        float[] values = getFloats(csv,0,SIZE-1);
        // read image and create input tensor
        TensorShape ts0 = new TensorShape(SIZE-1);// create label tensor
        Tensor t0 = new Tensor( Nd4j.create( values, indar_image, 'c' ), ts0 );
                    
          return new TensorPair(t0,t1);
    }

    private float getFloat(CSVRecord record, int  i) {
        String field = record.get(i);
        try {
            return Float.parseFloat(field);
        } catch(NumberFormatException e) {
            System.err.println(
                format("NumberFormatException: trying to read \"%s\" as float ",field)
            );
            return 0.0f;
        }
    }
    
    private float[] getFloats(CSVRecord record, int i0, int i1) {
        float[] result=new float[i1-i0];
        for(int i=0; i<i1-i0; i++) {
            result[i]=getFloat(record,i0+i);
        }
        return result;
    }

    /** returns the headers in order of appearance of the file. */
    private List<String> getHeaders(Map<String,Integer> map) {
        return map.entrySet().stream()
           .sorted((a,b)-> a.getValue().compareTo(b.getValue()))
           .map(e->e.getKey())
           .collect(toList());
    }
}
