package Recommender;

import Recommender.util.PearsonCoefficient;
import javafx.util.Pair;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.*;
import scala.collection.JavaConversions;
import scala.collection.mutable.WrappedArray;

import java.io.IOException;
import java.util.*;

public class movieRecommender {

    public static int currentUserId;
    public static double currentAvgUserRating;
    public static HashMap<Integer, Double> currentUserRatings;

    public static void main(String[] args) throws IOException {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("movie_recommender").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///d:spark_tmp").getOrCreate();

        Dataset<Row> ratingsData = spark.read().option("header", true).option("inferSchema", true).csv("src/main/resources/ml-100k/ratings.csv");

        ratingsData = ratingsData.drop("timestamp");
        ratingsData.createOrReplaceTempView("ratings");

        Dataset<Row> movieData = spark.read().option("inferSchema",true).option("header",true).csv("src/main/resources/ml-100k/movies.csv");
        movieData.createOrReplaceTempView("movies");

//        Dataset<Row> newUserList = spark.read().option("inferSchema",true).option("header",true).csv("src/main/resources/ml-100k/newUserList.csv");

        int moviesConsidered = 100; //should be that much greater than (pearsonMoviesConsidered+30), based on how many Matrix Factorisation recs you want
        // ie: if moviesConsidered = 100, pearsonMoviesConsidered = 20, then (100 - (20+30)) = 50 movies will be recommended by Matrix Factorisation algorithm
        int pearsonMoviesConsidered = 20;
        double penaltyMultiplier = 1.5; // if there's a negative genre rating, what multiplier to penalise it with
        double bonusMultiplier = 1.0; // if there's a positive genre rating, what multiplier to reward it with
        double closestMultiple = 0.001;
        int numUsers = 611;
        boolean showRecommendations = false;

        PearsonCoefficient.getUserMeans(spark, numUsers);

        List<Double> rmseList = new ArrayList<>();
        List<Double> mapeList = new ArrayList<>();

        for(int i=400;i<=500;i++)
        {
            int userId = i;

//            Dataset<Row> newUserList = spark.sql("select * from ratings where userId=" + userId);
            List<movieScorePair> scoring = getScores(spark, ratingsData, movieData, userId,
                    numUsers, moviesConsidered, pearsonMoviesConsidered, bonusMultiplier, penaltyMultiplier, closestMultiple);

            double N = currentUserRatings.size();
            double totalSquaredError = 0,totalPercentError = 0;
            if(showRecommendations)
            {
                System.out.println("\nUserId: " + userId + "\nRecommendations:");
            }
            for(movieScorePair p: scoring)
            {
                if(currentUserRatings.containsKey(p.movieId))
                {
                    double predictedScore = p.score;
                    double actualScore = (currentUserRatings.get(p.movieId));
                    double diff = predictedScore - actualScore;
                    totalSquaredError += (diff*diff);
                    totalPercentError += ((Math.abs(diff))/actualScore)*100;
//                    System.out.println(p.movie + ": assigned=" + p.score + ", original=" + currentUserRatings.get(p.movieId));
                }
                else if(showRecommendations)
                {
                    System.out.println(p.movie + ": " + p.score);
                }
            }

            double rmseScore = Math.sqrt(totalSquaredError/N);
            double mapeScore = totalPercentError/N;
            rmseList.add(rmseScore);
            mapeList.add(mapeScore);
            System.out.println("userId: "+ userId + ", rmseScore = " + rmseScore );
            System.out.println("userId: "+ userId + ", mapeScore = " + mapeScore );
        }

        Pair<Double, Double> rmseCTM = getCentralTendencyMeasures(rmseList);
        Pair<Double, Double> mapeCTM = getCentralTendencyMeasures(mapeList);
        double meanRMSE = rmseCTM.getKey();
        double meanMAPE = mapeCTM.getKey();
        double medianRMSE = rmseCTM.getValue();
        double medianMAPE = mapeCTM.getValue();

        System.out.println("Mean RMSE: " + meanRMSE);
        System.out.println("Mean MAPE: " + meanMAPE);
        System.out.println("Median RMSE: " + medianRMSE);
        System.out.println("Median MAPE: " + medianMAPE);
    }

    public static List<movieScorePair> getScores(SparkSession spark, Dataset<Row> ratingsData, Dataset<Row> movieData,
                                                 int userId, int numUsers, int moviesConsidered, int pearsonMoviesConsidered,
                                                 double bonusMultiplier, double penaltyMultiplier, double closestMultiple) throws IOException {
        currentUserId = userId;

        Dataset<Row> userMovieRatings = spark.sql("select * from ratings where userId=" + userId);

        // userRecs contains matrix factorisation recs
        Dataset<Row> userRecs = getRecs(ratingsData, userMovieRatings, moviesConsidered-pearsonMoviesConsidered-30);

        userMovieRatings = userMovieRatings.join(movieData,"movieId");
        //user movie ratings contains info about all movies rated by the user

//        System.out.println("userMovieRatings:");
//        userMovieRatings.show(false);


        // Getting "moviesConsidered" number of recommendations (at max.) in recs list
        List<Row> userRecsList = userRecs.takeAsList(moviesConsidered);
        // recs contains all the movie ids from the ALS recs
        ArrayList<Integer> recs = new ArrayList<>();
        for(Row r: userRecsList)
        {
            userId = r.getAs(0);
            int rec;
            Collection recCollection = JavaConversions.asJavaCollection(((WrappedArray) r.getAs("recommendations")).toList());;
            for (Row row : (Iterable<Row>) recCollection) {
                rec = (int) row.get(0);
                recs.add(rec);
            }
        }

        HashMap<Integer, Double> userRatings = new HashMap<>();
        // Creating genre rating
        Map<String, Pair<Double, Integer>> genreRating = new HashMap<>();
        double avgUserRating = PearsonCoefficient.userMeans[userId];
        List<Row> userMoviesList = userMovieRatings.takeAsList(30);

        // TODO recs could have repeated movieIds if some of the recommended movies and the user rated movies overlap
        for(Row r: userMoviesList)
        {
            int movieId = ((int) r.get(0));
            recs.add(movieId); // adding original user's movies to also be rated along with recs
            double rating = ((double) r.get(2));
            userRatings.put(movieId,rating);
            rating = (rating - avgUserRating);
            rating = (rating/avgUserRating)*100.00;
            String[] genres = r.get(4).toString().split("\\|");
            for(String genre: genres)
            {
                if(!genreRating.containsKey(genre))
                {
                    genreRating.put(genre, new Pair<Double, Integer>(rating,1));
                }
                else
                {
                    Pair<Double, Integer> p = genreRating.get(genre);
                    double prevRating = p.getKey();
                    int numOccurrences = p.getValue();
                    prevRating += rating;
                    numOccurrences++;
                    genreRating.put(genre,new Pair<Double, Integer>(prevRating,numOccurrences));
                }
            }
        }
        currentAvgUserRating = avgUserRating;
        currentUserRatings = userRatings;


        // TODO -Add Pearson recommended movie IDs to the recs list
        ArrayList<Integer> pearsonRecs = PearsonCoefficient.getPearsonRecs(spark, ratingsData,pearsonMoviesConsidered, userId, 10, numUsers);
        for(int movieId: pearsonRecs)
        {
//            System.out.println("adding movieId " + movieId);
            recs.add(movieId);
        }

//        System.out.println("Genre Rating:");
//        for (String genre : genreRating.keySet()) {
//            Pair<Double, Integer> p = genreRating.get(genre);
//            double prevRating = p.getKey();
//            int numOccurrences = p.getValue();
//            Double finalGenreRating = (prevRating/numOccurrences);
//            System.out.println("key: " + genre + " value: " + finalGenreRating);
//        }

        // collecting the recommended movies info from movieData
        Dataset<Row> recMovies, temp;
        recMovies = spark.sql("select * from movies where movieId=" + recs.get(0));
        for(int i=1;i<recs.size();i++)
        {
            int rec = recs.get(i);
            temp = spark.sql("select * from movies where movieId=" + rec);
            recMovies = recMovies.union(temp);
        }

        //scoring the movies from the collaborative filtering result(recs) using genreRating
        List<Row> recMoviesList = recMovies.takeAsList(moviesConsidered);
        List<movieScorePair> scoring = new ArrayList<>();
        for(Row row: recMoviesList)
        {
            double movieScore = 0;
            int movieId = (int)row.get(0);
            String movie = row.get(1).toString();
            String[] genres = row.get(2).toString().split("\\|");

            for(int i=0;i<genres.length;i++)
            {
                String genre = genres[i];
                if(genreRating.containsKey(genre))
                {
                    Pair<Double, Integer> p = genreRating.get(genre);
                    double prevRating = p.getKey();
                    int numOccurrences = p.getValue();
                    Double finalGenreRating = (prevRating/numOccurrences);
                    if(finalGenreRating < 0)
                    {
                        movieScore += (100+((penaltyMultiplier)*finalGenreRating));
                    }
                    else
                    {
                        movieScore += (100+((bonusMultiplier)*finalGenreRating));
                    }
                }
            }
            movieScore /= ((genres.length) * 100);
            movieScore *= avgUserRating;
            movieScore = roundTo(movieScore, closestMultiple);
            scoring.add(new movieScorePair(movieScore, movie, movieId));
        }

//        Collections.sort(scoring, (p1, p2) -> (int) (p2.score - p1.score));
        for(int i=0;i<scoring.size();i++)
        {
            for(int j=i+1;j<scoring.size();j++)
            {
                movieScorePair p1 = scoring.get(i);
                movieScorePair p2 = scoring.get(j);
                if(p2.score > p1.score)
                {
                    scoring.set(i,p2);
                    scoring.set(j,p1);
                }
            }
        }
        return scoring;
    }

    public static Dataset<Row> getRecs(Dataset<Row> ratingsData, Dataset<Row> newUserList, int moviesConsidered)
    {
        // matrix factorisation
        ALS als = new ALS();
        als.setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating");
        ALSModel model = als.fit(ratingsData);

        Dataset<Row> userRecs = model.recommendForUserSubset(newUserList,moviesConsidered);

        return userRecs;
    }

    public static double roundTo(double movieScore, double closestMultiple)
    {
        double diff1,diff2,lower,higher,a;
        int floor;
        a = (movieScore/closestMultiple);
        floor = (int) Math.floor(a);
        lower = closestMultiple*(floor);
        higher = closestMultiple*(floor+1);
        diff1 = movieScore-lower;
        diff2 = higher-movieScore;
        if(diff1 <= diff2)
        {
            return lower;
        }
        return higher;
    }

    public static Pair<Double, Double> getCentralTendencyMeasures(List<Double> rmseList)
    {
        for(int i=0;i<rmseList.size();i++)
        {
            for(int j=i+1;j< rmseList.size();j++)
            {
                double rmse1 = rmseList.get(i);
                double rmse2 = rmseList.get(j);
                if(rmse2 < rmse1)
                {
                    rmseList.set(i, rmse2);
                    rmseList.set(j, rmse1);
                }
            }
        }

        double mean=0, median;
        for(double score: rmseList)
        {
            mean += score;
        }
        mean /= (rmseList.size());
        int numScores = rmseList.size();
        if(numScores%2 == 0)
        {
            double a,b;
            a = rmseList.get((numScores/2)-1);
            b = rmseList.get((numScores/2));
            median = (a+b)/2;
        }
        else
        {
            median = rmseList.get((numScores/2));
        }
        
        return new Pair<>(mean, median);
    }
}

class movieScorePair {
    double score;
    String movie;
    int movieId;

    public movieScorePair(double score, String movie, int movieId)
    {
        this.score = score;
        this.movie = movie;
        this.movieId = movieId;
    }
}
