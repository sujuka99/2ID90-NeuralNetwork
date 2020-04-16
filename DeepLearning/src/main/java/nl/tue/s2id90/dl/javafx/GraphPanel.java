package nl.tue.s2id90.dl.javafx;

import java.util.List;
import javafx.scene.Node;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Data;
import javafx.scene.chart.XYChart.Series;
import static javafx.scene.input.KeyEvent.KEY_TYPED;
import javafx.scene.input.MouseEvent;
import lombok.Getter;
import nl.tue.s2id90.dl.javafx.GraphUpdater;

/**
 * Simple wrapper around LineChart that smoothes the input when the number of 
 * input gets large.
 * 
 * @author huub
 */
public class GraphPanel {
    
    int F = 1; // number of samples, needed for one data point in the graph
    
    private final MyLineChart<Number,Number> chart;
    private XYChart.Series<Number,Number> series;
    @Getter private String label;
    private final NumberAxis xAxis;
    private final NumberAxis yAxis;
    
    public GraphPanel(String label) {
        
        this.label = label;
        
        // initialize list of data points
        this.series = new Series<>();
        series.setName(label);
        
        // initialize chart
        this.chart = new MyLineChart(xAxis=new NumberAxis(), yAxis=new NumberAxis());
        chart.getData().add(new Series<>());
        chart.getData().get(0).setName(label);
        chart.setHorizontalZeroLineVisible(true);
        chart.setCreateSymbols(false);
        chart.getXAxis().setAnimated(false);
        chart.getYAxis().setAnimated(false);
        chart.getYAxis().setAutoRanging(true);
        chart.setAnimated(false);
        
        // install mouse handler that react on double mouse_clicks
        chart.addEventHandler(MouseEvent.MOUSE_CLICKED, e -> zoom(e));
        chart.addEventFilter(MouseEvent.MOUSE_CLICKED, e-> { if (e.getClickCount()!=2) e.consume(); });
        
        (new GraphUpdater(this::update)).start();
    }
    
    private void zoom(MouseEvent e) {
        System.err.println("keys=" + e);
        switch (e.getButton()) {
            case PRIMARY:
                yAxis.setAutoRanging(false);
                yAxis.setUpperBound(yAxis.getUpperBound() / 2);
                break;

            case SECONDARY:
                yAxis.setAutoRanging(false);
                yAxis.setUpperBound(2 * yAxis.getUpperBound());
                break;
            case MIDDLE:
                yAxis.setAutoRanging(true);
                break;
        }
    }
    
    public Node getNode() {
        return chart;
    }
    
    public void add(Number x, Number y) {
        series.getData().add(new Data<>(x,y));
    }
    
    // next unprocessed registered data point
    int nextDataPoint = 0;
    
    private void update() {
        if (chart.getData().isEmpty()) return;
        Series<Number, Number> chartSeries = chart.getData().get(0);
        
        // adapt filter size = the number of points to average
        int oldF = F;
//        while (nop >= M*F) F = F*50;
        F = Math.max(1, ((series.getData().size()/150)/10)*10);
        
        if (oldF==F) { // if filter size did not change, add points to series
            nextDataPoint = addPoints(series, nextDataPoint, F, chartSeries);
        } else {
            // create and fill a new series with averaged data
            Series<Number,Number> newSeries = new Series<>();
            nextDataPoint = addPoints( newSeries, 0, F, chartSeries);
            
            // replace series in chart by new series
            chart.getData().setAll(newSeries);
            newSeries.setName(label);
        } 
    }

    private int addPoints(Series<Number, Number> fromSeries, int nextPoint, int count, Series<Number,Number> toSeries) {
        List<Data<Number,Number>> fromList = fromSeries.getData();
        List<Data<Number,Number>> toList   = toSeries.getData();
        int nop = fromList.size();
        while (nextPoint+count-1 < nop) {
            int batch_index = fromList.get(nextPoint+count-1).getXValue().intValue();
            toList.add( new Data(batch_index, average(fromList, nextPoint, count)) );
            nextPoint += count;
        }
        return nextPoint;
    }

    private double average(List<Data<Number, Number>> fromList, int start, int count) {
        double average = 0;
        for (int i = start; i < start + count; i++) {
            average += fromList.get(i).getYValue().doubleValue();
        }
        return average / count;
    }
}