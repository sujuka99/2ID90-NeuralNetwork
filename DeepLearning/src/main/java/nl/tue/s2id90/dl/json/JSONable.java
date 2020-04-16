package nl.tue.s2id90.dl.json;

import org.json.simple.JSONObject;

/**
 *
 * @author huub
 */
public interface JSONable {
    /** returns a json representation of the implementing class object in the form 
     * of a JSONObject.
     * @return completed JSONObject
     */

    default public JSONObject json() {
        JSONObject jo = new JSONObject();
        jo.put("type",getClass().getSimpleName());
        return jo;
    }
}
