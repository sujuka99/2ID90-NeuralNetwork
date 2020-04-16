package nl.tue.s2id90.dl.javafx.widgets;

import java.io.IOException;
import java.net.URL;
import java.text.DecimalFormat;
import java.text.Format;
import java.util.ResourceBundle;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.StringProperty;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.layout.VBox;

public class SliderWidget extends VBox implements Initializable {

    @FXML private Label label;
    @FXML private Slider slider;
    @FXML private TextField valueField;
    /**
     * Initializes the controller class.
     * @param url
     * @param rb
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        DecimalFormat format = new DecimalFormat();
        format.setMaximumFractionDigits(6);
        valueField.textProperty().bindBidirectional(slider.valueProperty(), format);
    }    
    
    public SliderWidget(String label, Double value) {
        this();
        valueProperty().setValue(value);
        labelProperty().setValue(label);
    }

    public SliderWidget() {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/fxml/SliderWidget.fxml"));
        fxmlLoader.setRoot(this);
        fxmlLoader.setController(this);

        try {
            fxmlLoader.load();
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        }
    }
    
    public StringProperty labelProperty() {
        return label.textProperty();
    }
    
     public DoubleProperty valueProperty() {
        return slider.valueProperty();
    }
}