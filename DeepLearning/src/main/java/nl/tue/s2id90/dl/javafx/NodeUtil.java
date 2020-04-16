package nl.tue.s2id90.dl.javafx;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;

/**
 *
 * @author huub
 */
public class NodeUtil {
    public static TableView createInfoView(String label, Map<String,Object> infoMap) {
        if (label.equals("Model")) {
            return createModelInfoView(infoMap);
        } else {
            return createInfoView(infoMap);
        }
    }
    private static TableView createInfoView(Map<String,Object> infoMap) {
        Set<Entry<String, Object>> entrySet = infoMap.entrySet();
        
        TableView<Entry<String,Object>> table = new TableView<>();
        table.setItems(FXCollections.observableArrayList(entrySet));
        
        TableColumn<Entry<String,Object>,String> col1 = new TableColumn<>("Key");
        col1.setCellValueFactory(p -> new SimpleStringProperty(p.getValue().getKey()));
        TableColumn<Entry<String,Object>,String> col2 = new TableColumn<>("Value");
        col2.setCellValueFactory(p -> new SimpleObjectProperty(p.getValue().getValue()));
        
        table.getColumns().setAll(col1,col2);
        
        return table;
    }
    
    private static TableView createModelInfoView(Map<String,Object> infoMap) {
        Set<Entry<String, Object>> entrySet = infoMap.entrySet();
        
        TableView<Entry<String,Object>> table = new TableView<>();
        table.setItems(FXCollections.observableArrayList(entrySet));
        
        TableColumn<Entry<String,Object>,String> col1 = new TableColumn<>("Layer");
        col1.setCellValueFactory(p -> new SimpleStringProperty(p.getValue().getKey()));
        table.getColumns().add(col1);
        
        List<String> keys = Arrays.asList( "name", "input shape", "output shape", "activation");
        
        for(String key : keys) { 
            TableColumn<Entry<String,Object>,String> col = new TableColumn<>(key);
            col.setCellValueFactory(p -> new SimpleObjectProperty(getValueFromEntry(p.getValue(),key)));
            table.getColumns().add(col);
        }
        
        TableColumn<Entry<String,Object>,String> lastCol = new TableColumn<>("other");
        lastCol.setCellValueFactory(p -> new SimpleObjectProperty(getRemainingValuesFromEntry(p.getValue(),keys)));
        
        table.getColumns().add(lastCol);
        
        return table;
    }
    
    private static String getValueFromEntry(Entry<String,Object> entry, String name) {
        Map<String,Object> map = (Map<String,Object>) entry.getValue();
        return map.get(name)==null? "" : map.get(name).toString();
    }
    
    private static String getRemainingValuesFromEntry(Entry<String,Object> entry, List<String> keys) {
        Map<String,Object> map = (Map<String,Object>) entry.getValue();
        Set<String> rest = new HashSet(map.keySet());
        rest.removeAll(keys);
        return rest.stream()
                   .map(key->key+": "+map.get(key))
                   .collect(Collectors.joining(", "));
    }
}
