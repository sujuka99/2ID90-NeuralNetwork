package nl.tue.s2id90.dl.javafx;

import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

/**
 *
 * @author huub
 */
public class FXGUI extends FXBase {
    static FXGUI gui;
    TabPane tabPane = new TabPane();
    HBox statusBar = new HBox(3);
    Stage stage;
    
    @Override public void init() {
        countDown(); // you have to call this!
    }

    @Override public void start(Stage primaryStage) {
        gui = this;
        this.stage = primaryStage;
        
        statusBar.setStyle("-fx-font-size: 10;");
        VBox.setVgrow(statusBar, Priority.NEVER);
        VBox.setVgrow(tabPane, Priority.ALWAYS);
        
        VBox  vbox= new VBox(4,statusBar,tabPane);
        vbox.setMinWidth(vbox.USE_COMPUTED_SIZE);
        Scene scene = new Scene(vbox, 300, 250);
        //tabPane.getTabs().add(new Tab("About",new Text("hello world")));
        primaryStage.setScene(scene);
        primaryStage.setOnCloseRequest(e -> System.exit(0));
        primaryStage.show();
        
        countDown();    // you have to do this!
    }
    
    public void setTitle(String title) {
        Platform.runLater(()->stage.setTitle(title));
    }
    
    public static boolean isAvailable() {
        return gui!=null;
    }
    
    public static FXGUI getSingleton() {
        if (gui==null) {
            FXBase.launchFXAndWait(FXGUI.class);
        }
        return gui;
    }
    
    /** Adds a tab with the given label and node.
     * @param label label of the tab
     * @param node  widget shown under the tab
     */
    public void addTab(String label, Node node) {
        Platform.runLater(()-> tabPane.getTabs().add(new Tab(label,node)));
    }
    
    /** Adds a tab with the given label and node at the given position.
     * @param i position to add the tab
     * @param label label of the tab
     * @param node  widget shown under the tab
     */
    public void addTab(int i, String label, Node node) {
        Platform.runLater(()-> tabPane.getTabs().add(i,new Tab(label,node)));
    }
    
    /** convenience method for adding a tab with a GraphPanel.
     * @param gp panel, will be added with label gp.getLabel()
     */
    public void addTab(GraphPanel gp) {
        addTab(gp.getLabel(), gp.getNode()); 
    };
    
    /** convenience method for adding a tab with a GraphPanel.
     * @param activations Activations panel, will be added with label "activations".
     */
    public void addTab(Activations activations) {
        addTab("activations", activations); 
    };
    
    /** returns statusBar of GUI. **/
    public void addStatus(Node ... children) {
        Platform.runLater(()->statusBar.getChildren().addAll(children));
    }
}
