package Recommender;

import Recommender.util.PearsonCoefficient;
import javafx.util.Pair;

import java.io.IOException;
import java.util.ArrayList;

public class PearsonRecommendations {

    public static void main(String[] args) throws IOException {

        ArrayList<Pair<Integer, Double>> topK = PearsonCoefficient.getTopKUtility(10,611, 611);
        for(Pair<Integer, Double> p : topK)
        {
            int userId = p.getKey();
            Double coeff = p.getValue();
            System.out.println(userId + ": " + coeff);
        }

    }

}
