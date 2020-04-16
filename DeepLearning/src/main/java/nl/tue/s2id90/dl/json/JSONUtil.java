package nl.tue.s2id90.dl.json;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.string.NDArrayStrings;

/**
 *
 * @author huub
 */
public class JSONUtil {
    
    public static JSONArray toJson(INDArray array) {
        final int PRECISION=17;
        
        // handle special case when array is actually a scalar
        String json;
        if (array.isScalar())
            json=String.format("[%."+PRECISION+"f]",array.getDouble(0));
        else
            json= new NDArrayStrings(",",PRECISION).format(array);
        
        try {
            return (JSONArray) new JSONParser().parse(json);
        } catch (ParseException ex) {
            return null;
        }
    }
    
    static public <T> List<T> toList(JSONObject jo, String key, Class<T> clazz) {
        JSONArray a = (JSONArray)jo.get(key);
        return toList(a,clazz);
    }
     
    static public <T> List<T> toList(JSONArray a, Class<T> clazz) {
        List<T> list=new ArrayList<>();
        for(int i=0;i<a.size();i++) {
            list.add((T)a.get(i));
        }
        return list;
    }
    
    public static String getString(JSONObject jo, String key) {
        return (String)jo.get(key);
    }
    
    public static int getInt(JSONObject jo, String key) {
        return ((Long)jo.get(key)).intValue();
    }
    
    public static double getDouble(JSONObject jo, String key) {
        return ((Double)jo.get(key));
    }
   
    
    //<editor-fold defaultstate="collapsed" desc="formatting"> 
    public static String format(String json) {
        return compactify(formatJSONStr(json, 4));
    }
    
    private static String compactify(String json) {
        List<String> keys = Arrays.asList("input_shape","output_shape","shape","bias","weights","loss","activation");
        StringBuilder result = new StringBuilder();
        int newLine = 0;
        List<String> lines = Arrays.asList(json.split("\n"));
        for (String line : lines) {
            String trimmedLine = line.trim();
            if (keys.stream().anyMatch(     // if starts with one of the keys surrounded by double quotes
                    key -> trimmedLine.startsWith("\""+key+"\""))
                    ) {
                result.append(newLine==0 ? "\n" + line : trimmedLine);
                newLine ++;
            } else if (newLine!=0 && trimmedLine.startsWith("]")) {
                newLine--;
                result.append(line.trim());
            } else if (newLine!=0 && trimmedLine.startsWith("}")) {
                newLine--;
                result.append(trimmedLine);
            } else {
                result.append(newLine==0 ? "\n" + line : trimmedLine);
            }
        }
        return result.toString();
    }
    
    /* next method is copied from internet, seems to work; strange usage of continue though. */
    private static String formatJSONStr(final String json_str, final int indent_width) {
        final char[] chars = json_str.toCharArray();
        final String newline = System.lineSeparator();
        
        StringBuilder ret = new StringBuilder("");
        boolean begin_quotes = false;
        
        for (int i = 0, indent = 0; i < chars.length; i++) {
            char c = chars[i];
            
            if (c == '\"') {
                ret.append(c);
                begin_quotes = !begin_quotes;
                continue;
            }
            
            if (!begin_quotes) {
                switch (c) {
                    case '{':
                    case '[':
                        ret.append(c).append(newline).append(String.format("%" + (indent += indent_width) + "s", ""));
                        continue;
                    case '}':
                    case ']':
                        ret.append(newline).append((indent -= indent_width) > 0 ? String.format("%" + indent + "s", "") : "").append(c);
                        continue;
                    case ':':
                        ret.append(c).append(" ");
                        continue;
                    case ',':
                        ret.append(c).append(newline).append(indent > 0 ? String.format("%" + indent + "s", "") : "");
                        continue;
                    default:
                        if (Character.isWhitespace(c)) {
                            continue;
                        }
                }
            }
            
            ret.append(c).append(c == '\\' ? "" + chars[++i] : "");
        }
        
        return ret.toString();
    }
    //</editor-fold>
}
