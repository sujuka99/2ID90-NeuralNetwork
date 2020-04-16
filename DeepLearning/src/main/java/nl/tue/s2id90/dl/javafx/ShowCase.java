package nl.tue.s2id90.dl.javafx;

import static java.lang.String.format;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import static java.util.stream.Collectors.toList;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.Text;
import nl.tue.s2id90.dl.NN.Model;
import nl.tue.s2id90.dl.NN.tensor.Tensor;
import nl.tue.s2id90.dl.NN.tensor.TensorPair;
import nl.tue.s2id90.dl.Nd4jUtil;
import static nl.tue.s2id90.dl.javafx.Images.getFXImage;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Shows image, label and scores of a list of Tensor_Pairs.
 * @author huub
 */
public class ShowCase {
    Function<Integer,String> toString;
    private final FlowPane flowPane;    // javafx node containing images and the like.
    public ShowCase(Function<Integer,String> toString) {
        this.toString = toString;
        
        /* initializes main layout: FlowPane */
        flowPane = new FlowPane();
        flowPane.setVgap(15);
        flowPane.setHgap(15);
    }
    
    /** 
     * @return  javafx node containing this showCase.
     */
    public Node getNode() {
        ScrollPane pane = new ScrollPane(flowPane);
        pane.setFitToWidth(true);
        
        // make flowPane as wide as possible without need for scrolling horizontally
        //flowPane.prefWrapLengthProperty().bind(pane.widthProperty());
        
        return pane;
    }
    
    /** adds gui representations (Items) of the tensor pairs to the flow pane.
     * @param pairs list of tensor pairs
     **/
    public void setItems(List<TensorPair> pairs) {
        List<TensorPair> sortedPairs = new ArrayList(pairs); // sort by label
        Collections.sort(sortedPairs, 
                (o1,o2)-> Integer.compare(
                        label(o1.model_output.getValues()),
                        label(o2.model_output.getValues())
                )
        );
        
        /** create list of images **/
        Platform.runLater( ()->
            flowPane.getChildren().setAll(
                pairs.stream()
                .map(tp-> new Item(tp))
                .collect(toList())
                )
        );
    }  
    
    public void update(Model model) {
        List<Node> items = flowPane.getChildren();
        
        for(int i=0;i<items.size(); i++) {
            Item item = (Item)items.get(i);
            item.update(model);
        }
    }
    
    /** converts tensor to a javafx image. */
    private Image image(Tensor t) {
        return getFXImage(Images.image_from_tensor_3d(t));
    }
    
    /** computes label id. */
    private int label(INDArray array) {
        return Nd4jUtil.argMax(array);
    }
    
    private List<Float> scores(Model model, TensorPair tp) {
        if (model!=null) {
            return scores(model.inference(tp.model_input).getValues());
        } else {
            return Collections.emptyList();
        }
    }
    
    /** converts class scores to list of floats. */
    private List<Float> scores(INDArray array) {
        List<Float> result = new ArrayList<>();
        INDArray a = array.reshape(1,array.length());
        for(int i=0;i<a.length();i++) {
            result.add(a.getFloat(i));
        }
        return result;
    }
    
    /** widget for showing image, label and label scores. */
    class Item extends VBox {
        int label;
        HBox histogram;
        TensorPair tp;
        public Item(TensorPair tp) {
            this.tp = tp;
            Image image = image(tp.model_input);
            int label=label(tp.model_output.getValues());
            List<Float> scores = Collections.emptyList(); // no scores yet
            this.label = label;
            String labelString = format("%d - %s", label, toString.apply(label) );
            getChildren().addAll(
                new ImageView(image), 
                histogram = new HBox(),
                new Text(labelString)
            );
            this.setSpacing(2);
            rectangles(scores);
        }
        
        void setScores(List<Float> scores) {
            rectangles(scores);
        }
        
        /** creates horizontal histogram. */
        private void rectangles(List<Float> scores) {
            List<Rectangle> rectangles = new ArrayList<>();
            for (int i = 0; i < scores.size(); i = i + 1) {
                Float s = scores.get(i);
                rectangles.add(new Rectangle(8, Math.max(1, 25 * s), i == label ? Color.GREEN : Color.RED));
            }
            Platform.runLater( () ->
                histogram.getChildren().setAll(rectangles.toArray(new Rectangle[0]))
            );
           histogram.setSpacing(2);
        }

        private void update(Model model) {
            List<Float> scores = scores(model.inference(tp.model_input).getValues());
            this.setScores(scores);
    }
    }
    
}
