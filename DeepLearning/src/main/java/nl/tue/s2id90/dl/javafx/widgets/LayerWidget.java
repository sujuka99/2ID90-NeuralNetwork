package nl.tue.s2id90.dl.javafx.widgets;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.geometry.Insets;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import nl.tue.s2id90.dl.experiment.ActivationData;
import nl.tue.s2id90.dl.javafx.ImageCreator;
import nl.tue.s2id90.dl.javafx.Images;

public class LayerWidget extends VBox implements Initializable {

    @FXML private Label nameLabel;
    ImageCreator image_creator= new ImageCreator(2, 5, 4);

    /**
     * Initializes the controller class.
     * @param url
     * @param rb
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        nameLabel.setText("");
        setPrefWidth(USE_COMPUTED_SIZE);
        setPrefHeight(USE_COMPUTED_SIZE);
        setMaxWidth(Double.MAX_VALUE);
        setStyle("-fx-background-color:gray;");
        setPadding(new Insets(0,0,0,10)); // left = 10
    }
                
    public void set(final ActivationData data, final int index, final int sample) {
        widthProperty().addListener((source,oldValue,newValue) -> {
            double max_width = newValue.doubleValue()-10;
            if (max_width<=0) return;
            nameLabel.setText(data.names.get(index));
            BufferedImage image = image_creator.create_image(
                    data.tensors.get(index),
                    data.show_values.get(index),
                    (int)max_width, sample);
            Image fxImage = Images.getFXImage(image);
            ImageView iv = new ImageView(fxImage);
            if (getChildren().size()>1) {
                getChildren().set(1,iv);
            } else {
                getChildren().add(iv);
            }
            iv.setPreserveRatio(true);
            VBox.setVgrow(iv, Priority.ALWAYS);
        });
    }    

    public LayerWidget() {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/fxml/layerwidget.fxml"));
        fxmlLoader.setRoot(this);
        fxmlLoader.setController(this);

        try {
            fxmlLoader.load();
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        }
    }
}