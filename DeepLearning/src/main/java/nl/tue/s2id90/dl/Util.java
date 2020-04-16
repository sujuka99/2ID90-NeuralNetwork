package nl.tue.s2id90.dl;

import static java.lang.String.format;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;
import java.util.stream.IntStream;

/**
 *
 * @author huub
 */
public class Util {
    
    /**
     * @param a integer array 
     * @return string representation of a.
     */
    public static String arrayToString(int[] a) {
        return IntStream.of(a).boxed().collect(toList()).toString();
    }
    
    public static <T> List<T> pickRandom(List<T> list, int n) {
        if (n > list.size()) {
            throw new IllegalArgumentException("not enough elements");
        }
        Random random = new Random();
        return IntStream
                .generate(() -> random.nextInt(list.size()))
                .distinct()
                .limit(n)
                .mapToObj(list::get)
                .collect(Collectors.toList());
    }
    
    public static void main(String[] args) {
        List<Integer> a = Arrays.asList(1,2,3,4,5);
        System.err.println(pickRandom(a, 1));
        System.err.println(pickRandom(a, 1));
        System.err.println(pickRandom(a, 1));
        System.err.println(pickRandom(a, 1));
    }
    
    public static String toString(Map<String,Object> infoMap) {    
        return infoMap.entrySet().stream()
            .map(e->format("%-20s: %s",e.getKey(), e.getValue()))
            .collect(joining("\n"));
    }
}
