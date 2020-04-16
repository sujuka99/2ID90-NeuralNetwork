package nl.tue.s2id90.dl.javafx;

import java.net.URL;
import java.util.ResourceBundle;
import java.util.function.DoubleFunction;
import javafx.beans.property.ReadOnlyDoubleProperty;
import javafx.beans.property.ReadOnlyDoubleWrapper;
import javafx.fxml.Initializable;
import javafx.scene.control.Slider;
import javafx.util.StringConverter;

public class FunctionalSlider extends Slider implements Initializable {

    private ReadOnlyDoubleWrapper functionValue = new ReadOnlyDoubleWrapper();
    
    public FunctionalSlider() {
        this(d->d);
    }

    public FunctionalSlider(DoubleFunction<Double> function) {
        _setFunction(function);
    }
    
    public void setFunction(DoubleFunction<Double> f) { _setFunction(f); }
    
    private void _setFunction(DoubleFunction<Double> f) { 
        valueProperty().addListener(
            o -> functionValue.set(f.apply(getValue()))
        );

        setLabelFormatter(new StringConverter<Double>() {
            @Override public String toString(Double x) {
                return String.format("%1.0f", f.apply(x));
            }

            @Override public Double fromString(String s) { return null; }
        });
    }

    public double getFunctionValue() { return functionValue.get(); }

    public ReadOnlyDoubleProperty functionValueProperty() {
        return functionValue.getReadOnlyProperty();
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) { }
    
    public static void main(String[] args) {
        FunctionalSlider fs = new FunctionalSlider(d->d);
    }
}
