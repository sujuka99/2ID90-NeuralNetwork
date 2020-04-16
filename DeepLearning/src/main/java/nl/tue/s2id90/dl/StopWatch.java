package nl.tue.s2id90.dl;

/**
 *
 * @author huub
 */
public class StopWatch {
    
    public StopWatch(boolean startImmediately) {
        if (startImmediately) reset();
    }
    Long startTime;
    private void reset() { startTime = System.nanoTime(); }
    
    /** starts measuring elapsed time. **/
    public void start() { reset();}
    
    /** @return elapsed time in nano seconds since last call to start(). **/
    public long get() { return System.nanoTime()-startTime; }
    
    /** return true iff elapsed time is more than t seconds. **/
    public boolean elapsedSecondsMoreThan(int t) {
        return get()/1E9 > t;
    }
}
