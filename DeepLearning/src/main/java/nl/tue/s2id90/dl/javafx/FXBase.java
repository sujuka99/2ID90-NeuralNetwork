package nl.tue.s2id90.dl.javafx;

import java.util.concurrent.CountDownLatch;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.application.Application;
import javafx.application.Platform;

/**
 * Launches javafx application and waits till FX toolkit is properly initialized.
 * @author huub
 */
public abstract class FXBase extends Application {
    private static final CountDownLatch LATCH = new CountDownLatch(2);

    void countDown() { 
        LATCH.countDown();  
    }

    public static void awaitFXToolkit() throws InterruptedException {
        LATCH.await();
    }

    /**
     * Starts javafx environment and pops up a frame
     */
    public static <T extends FXBase> void launchFXAndWait(Class<T> clazz) {
        Platform.setImplicitExit(false);
        new Thread(() -> Application.launch(clazz)).start();
        try {
            T.awaitFXToolkit();
        } catch (InterruptedException ex) {
            Logger.getLogger(FXBase.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
