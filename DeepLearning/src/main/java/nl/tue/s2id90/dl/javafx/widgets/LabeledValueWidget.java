package nl.tue.s2id90.dl.javafx.widgets;

import java.io.IOException;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.ResourceBundle;
import javafx.beans.property.StringProperty;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;

public class LabeledValueWidget extends HBox implements Initializable {

    @FXML private Label label;
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
    }    
    
    public LabeledValueWidget(String label, String value, double minimumWidthLabel) {
        this(label,value);
        this.setMinWidth(minimumWidthLabel);
    }
    
    public LabeledValueWidget(String label, String value) {
        this();
        valueField.textProperty().setValue(value);
        labelProperty().setValue(label);
    }

    public LabeledValueWidget() {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/fxml/LabeledValueWidget.fxml"));
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
    
     public StringProperty valueProperty() {
        return valueField.textProperty();
    }
}