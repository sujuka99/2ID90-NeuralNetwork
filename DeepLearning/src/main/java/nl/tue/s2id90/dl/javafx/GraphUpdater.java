package nl.tue.s2id90.dl.javafx;

import javafx.animation.AnimationTimer;

/**
 * Encapsulates a method that is called once per frame.
 * @author huub
 */
public class GraphUpdater extends AnimationTimer {

    private final Runnable method;

    public GraphUpdater(Runnable r) {
        this.method = r;
    }
    
    @Override
    public void handle(long now) {
        method.run();
    }
    
}
