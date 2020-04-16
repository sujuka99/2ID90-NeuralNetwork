package nl.tue.s2id90.dl.input;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import static java.util.stream.Collectors.toList;
import java.util.stream.IntStream;
import java.util.zip.GZIPOutputStream;
import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritablePixelFormat;
import javafx.scene.paint.Color;
import javafx.util.Pair;

/**
 *
 * @author huub
 */
public class PrimitivesDataWriter {
    final private static ImageGenerator GENERATOR= new ImageGenerator(
            true,         // square
            true,         // circle
            true,        // triangle
            true,        // rotated
            20180102,     // random seed
            28,           // image size
            0,            // noi,  not used if only nextImage() is called
            Color.BLACK,  // bgColor
            Color.WHITE,  // fgcolor
            false,         // anti-aliased
            null);        // Random generator (not used???)

    final static private int IDX3 = 2051;
    final static private int IDX1 = 2049;
    
    public PrimitivesDataWriter(String file, int samples) {
        List<Pair<Byte, Image>> data = IntStream.range(0, samples)
                    .mapToObj(i->GENERATOR.nextImage())
                    .map(p->new Pair<>(toByte(p.getKey()),p.getValue()))
                    .collect(toList()); 
        List<Byte> labels = data.stream().map(p->p.getKey()).collect(Collectors.toList());
        List<Image  > images = data.stream().map(p->p.getValue()).collect(Collectors.toList());
        
        writeData(file+"-images-idx3-ubyte.gz", IDX3, images, label->toByte(label));
        writeData(file+"-labels-idx1-ubyte.gz", IDX1, labels, image->toByte(image));
    }
    
    private <T> void writeData(String file, int idx, List<T> data, Function<T,byte[]> convert)  {// open ByteStream
        int size = data.size();
        try (
            GZIPOutputStream zip = new GZIPOutputStream(new FileOutputStream(file))
        ) {        
            zip.write(ByteBuffer.allocate(4).putInt(idx).array());
            zip.write(ByteBuffer.allocate(4).putInt(size).array());
            if (idx==IDX3) {
                zip.write(ByteBuffer.allocate(4).putInt(28).array());    // rows
                zip.write(ByteBuffer.allocate(4).putInt(28).array());    // columns
            }
            data.stream().forEach(element -> {
                byte[] b = convert.apply(element);
                try {
                    zip.write(b);
                    System.err.print(".");
                } catch (IOException ex) {
                    Logger.getLogger(PrimitivesDataWriter.class.getName()).log(Level.SEVERE, null, ex);
                }
            });
        } catch (IOException ex) {
            Logger.getLogger(PrimitivesDataWriter.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.err.print("\n\n\n\n");
    }
    
    private Byte toByte(String t) { 
        switch (t) {
            case "S": return 0;
            case "C": return 1;
            default:  return 2;
        }
    }
    
    private byte[] toByte(Byte b) { return new byte[] { b }; }
    
    private byte[] toByte(Image image) { return values(image); }
    
    private static byte[] values(Image image) {
        int w=GENERATOR.getSize(), h=GENERATOR.getSize();
        byte[] rgba = new byte[4*w*h]; 
        PixelReader pr = image.getPixelReader(); 
        WritablePixelFormat<ByteBuffer> format = PixelFormat.getByteBgraInstance();
        pr.getPixels(0, 0, w, h, format, rgba, 0, 4*w); 
        
        ByteBuffer buf = ByteBuffer.allocate(w*h);
        byte[] grays = new byte[w*h];
        for(int i=0;i<grays.length;i++) {
            byte b = rgba[4*i], g=rgba[4*i+1], r=rgba[4*i+2], a=rgba[4*i+3];
            //grays[i]= r;   // assuming that color in image was a gray color with r=g=b
            Float gray = (float)Color.rgb(r&0xFF, g&0xFF, b&0xFF).grayscale().getRed();
            grays[i] = (byte)(gray*255);
            buf.put(grays[i]);
        }
        return buf.array();
    }
    
    public static void main(String[] args) throws IOException {
        new PrimitivesDataWriter("train", 6000);
        new PrimitivesDataWriter("t10k", 1000);
        System.err.println("finished");
    }
}