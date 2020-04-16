package nl.tue.s2id90.dl.input;

import static java.lang.String.format;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import javafx.application.Platform;
import javafx.beans.property.SimpleObjectProperty;
import javafx.embed.swing.JFXPanel;
import javafx.geometry.Insets;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Polygon;
import javafx.scene.shape.Rectangle;
import javafx.scene.shape.Shape;
import javafx.scene.transform.Rotate;
import javafx.util.Pair;
import lombok.Builder;
import lombok.Getter;
import lombok.NonNull;

/**
 *
 * @author huub
 */
@Builder
public class ImageGenerator {
    static { // initialize javafx platform
        new JFXPanel();
    }
    @Builder.Default @Getter private boolean square=true;
    @Builder.Default @Getter private boolean circle=true;
    @Builder.Default @Getter private boolean triangle=false;
    @Builder.Default @Getter private boolean rotated=false;
    @Builder.Default @Getter private int seed=110861;
    @Builder.Default @Getter private int size=28;
    @Builder.Default @Getter private int noi=1024;
    @Builder.Default @Getter private Color bgColor=Color.WHITE;
    @Builder.Default @Getter private Color fgColor=Color.BLACK;
    @Builder.Default @Getter private boolean antialias=false;
        
    Random rnd;
    
    public Map<String,List<Image>> generate() {
        Map<String,List<Image>> result = new HashMap<>();
        
        for(int i=0;i<noi;i++) {
            System.err.print(format("(%d)%s",i,(i%100==0?"\n":"")));
            Pair<String,Image> entry = nextImage();
            String type = entry.getKey();
            Image image = entry.getValue();
            
            List<Image> list = result.getOrDefault(type, new ArrayList<>());
            list.add(image);
            result.putIfAbsent(type, list);
        }
        return result;
    }
    
    public Pair<String,Image> nextImage() {
        try {
            if (rnd == null) {
                rnd = new Random(seed);
            }
            String type = nextType();

            Image image = generateImage(type);
            return new Pair<>(type, image);
        } catch (InterruptedException e) {
            return null;
        }
    }
    
    private String nextType() {
        List<String> types = new ArrayList();
        if (square  ) types.add("S");
        if (circle  ) types.add("C");
        if (triangle) types.add("T");
        int noTypes = types.size();        
        if (noTypes<=0) throw new IllegalArgumentException(format("noTypes=%2d should be > 0",noTypes));
        
        return types.get(rnd.nextInt(noTypes));   
    }

    private Image generateImage(String type) throws InterruptedException {
        try {
        WritableImage image = new WritableImage(size, size);
        image.getPixelWriter().setColor(size/2, size/2, Color.CYAN);
        
        final Pane pane = new Pane();
        pane.setMinSize(size, size);
        pane.setPrefSize(size, size);
        pane.setMaxSize(size,size);
        pane.setBackground(new Background(new BackgroundFill(bgColor, CornerRadii.EMPTY, Insets.EMPTY)));
        
        Shape shape=null;
        
        int d=10+rnd.nextInt(6);   // diameter
        int d2 = d/2;
        int x = d2 + rnd.nextInt(size-d);  
        int y = d2 + rnd.nextInt(size-d);
        System.err.format("\n%s(x-d2,x,x+d2)=(%3d,%3d,%3d)",type,x-d2,x,x+d2);
        System.err.format("%s(y-d2,y,y+d2)=(%3d,%3d,%3d)\n",type,y-d2,y,y+d2);
        
        switch(type) {
            case "S"   : shape = new Rectangle(snap(x-d2),snap(y-d2),d, d); break;
            case "C"   : shape = new Circle(snap(x),snap(y),d2);      break;
            case "T"   : shape = new Polygon(x-d2, y-d2,
                                             x+d2, y-d2,
                                             x, y + d2
                                    ); break;
        }
        if (rotated) {
            double angle=rnd.nextDouble();
            shape.getTransforms().add(new Rotate(angle*360.0, x, y));
        }
        shape.setStroke(fgColor);
        shape.setFill(Color.RED);
        shape.setSmooth(antialias);
        
        shape.setClip(pane.getShape());
        pane.getChildren().add(shape);
        
        return snapshot(pane);
        } catch(Exception e) {
            System.err.println(e);
            return null;
        }
    }
    
    private double snap(double y) {
      return ((int) y) + .5;
    }
    
     private WritableImage snapshot(@NonNull final Parent chartContainer) throws InterruptedException {
      final CountDownLatch latch = new CountDownLatch(1);
      // render the chart in an offscreen scene (scene is used to allow css processing) and snapshot it to an image.
      // the snapshot is done in runlater as it must occur on the javafx application thread.
      final SimpleObjectProperty<WritableImage> imageProperty = new SimpleObjectProperty();
      Platform.runLater(() -> {
          Scene snapshotScene = new Scene(chartContainer);
          final SnapshotParameters params = new SnapshotParameters();
          params.setFill(bgColor);
          chartContainer.snapshot(
                  result -> { 
                      imageProperty.setValue(result.getImage());
                      latch.countDown();
                      return null;
                  },
                  params,
                  null
          );
      });

      latch.await();  // wait for latch to get to zero!
      
      return imageProperty.get();
    }
}
