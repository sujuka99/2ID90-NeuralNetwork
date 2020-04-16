package nl.tue.s2id90.dl.javafx;

import nl.tue.s2id90.dl.javafx.widgets.LayerWidget;
import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import javafx.application.Platform;
import javafx.collections.FXCollections;

import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import nl.tue.s2id90.dl.experiment.ActivationData;

public class Activations extends StackPane implements Initializable {

    @FXML private VBox activationVBox;
    @FXML private ChoiceBox<ActivationData> batchesChoiceBox;
    @FXML private ComboBox<Integer> activationsComboBox;
    ScrollPane pane;
    /**
     * Initializes the controller class.
     * @param url
     * @param rb
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        batchesChoiceBox.getSelectionModel().selectedItemProperty()
                .addListener((observable, oldValue, newValue) -> {
                    selectActivation(newValue);
        });
        
        activationsComboBox.getSelectionModel().selectedItemProperty()
                .addListener((observable, oldValue, newValue) -> {
                        selectSample(newValue);
        });
    }    

    public Activations() {
        FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/fxml/activations.fxml"));
        fxmlLoader.setRoot(this);
        fxmlLoader.setController(this);

        try {
            fxmlLoader.load();
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        }
    }
    
    public void add(ActivationData activation) { // add choicebox item in javafx thread
        Platform.runLater(() -> batchesChoiceBox.getItems().add(activation));
    }
    
    private void selectActivation(ActivationData data) {
        if (data==null) return;
        int noSamples = (int)data.tensors.get( 0 ).getValues().shape()[0];
        activationsComboBox.setItems(
            FXCollections.observableArrayList(
                IntStream.range(0, noSamples).boxed().collect(Collectors.toList())
            )
        );
        
        activationsComboBox.getSelectionModel().select(0);
    }
     
    private void selectSample(Integer index) {
        if (index==null) return;
        
        ActivationData data = batchesChoiceBox.getSelectionModel().getSelectedItem();
        
        // remove current layout
        activationVBox.getChildren().clear();
        
        // do something for every layer
        for (int i=0; i< data.names.size(); i++) {
            LayerWidget layerWidget = new LayerWidget();
            activationVBox.getChildren().add(layerWidget);
            layerWidget.set(data,i,index);
        }
        activationVBox.layout();
    }
}