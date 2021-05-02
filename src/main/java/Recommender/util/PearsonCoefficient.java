package Recommender.util;

import javafx.util.Pair;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class PearsonCoefficient {

    public static Double[] userMeans;

    public static void main(String[] args) throws IOException {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("movieRecommender").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///d:spark_tmp").getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true)
                .option("inferSchema", true).csv("src/main/resources/ml-100k/ratings.csv");

        createUserMeanCSV(dataset, spark);

        int numUsers = 612;
        createPearsonCoeffsCSV(dataset, spark, numUsers);

        int topKForUser = 612;
        getTopK(10, topKForUser);
    }

    public static ArrayList<Long> getPearsonRecs(int numRecs, int userId, int kNeighbours, int numUsers) throws IOException {
        List<Pair<Integer, Double>> topNeighbours = getTopK(kNeighbours, userId);
        ArrayList<Long> recMovieIds = new ArrayList<>();



        return recMovieIds;
    }

    public static ArrayList<Pair<Integer, Double>> getTopKUtility(int k, int user, int numUsers) throws IOException {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("movieRecommender").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///d:spark_tmp").getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true)
                .option("inferSchema", true).csv("src/main/resources/ml-100k/ratings.csv");

        createUserMeanCSV(dataset, spark);

        createPearsonCoeffsCSV(dataset, spark, numUsers);

        ArrayList<Pair<Integer, Double>> topK = getTopK(k, user);
        return topK;
    }

    public static ArrayList<Pair<Integer, Double>> getTopK(int k, int user) throws FileNotFoundException {
        File f = new File("src/main/resources/ml-100k/pearsonCoeffs.csv");
        Scanner reader = new Scanner(f);
        TreeMap<Double, Integer> coeffMap = new TreeMap<>();
        reader.nextLine();

        while(reader.hasNextLine())
        {
            String[] row = reader.nextLine().split(",");
            int userId = 0;
            try
            {
                userId = Integer.parseInt(row[0]);
            }
            catch(Exception e)
            {
                System.out.println(row[0] + "couldn't be parsed");
                System.out.println(e);
            }
            double coeff = Double.parseDouble(row[user]);
            coeffMap.put(coeff, userId);
        }

        ArrayList<Pair<Integer, Double>> topNeighbours = new ArrayList<>();
        NavigableSet set = coeffMap.descendingKeySet();
        for(Object c: set)
        {
            double coeff = (Double)c;
            int userId = coeffMap.get(coeff);
//            System.out.println(userId + ": " + coeff);
            topNeighbours.add(new Pair<Integer, Double>(userId, coeff));
        }

        return topNeighbours;
    }

    public static void createPearsonCoeffsCSV(Dataset<Row> dataset, SparkSession spark, int numUsers) throws IOException {
        HashMap<Long, Double>[] userRatings = getUserRatings(dataset, numUsers);
        Double[] userMean = getUserMeans(spark, numUsers);
        Double[][] coeffs = new Double[numUsers+1][numUsers+1];

        for(int i=1;i<=numUsers;i++)
        {
            for(int j=i;j<=numUsers;j++)
            {
                if(i != j)
                {
                    double coeff = 1;
                    List<Long> commonMovies = new ArrayList<>();
                    for(Map.Entry elem: userRatings[i].entrySet())
                    {
                        long movieId = (Long)elem.getKey();
                        if(userRatings[j].containsKey(movieId))
                        {
                            commonMovies.add(movieId);
                        }
                    }

                    double dev1 = 0, dev2 = 0, denominator=1,numerator=0;
                    double user1Mean = userMean[i], user2Mean = userMean[j];

                    if(!commonMovies.isEmpty())
                    {
                        for(long movieId: commonMovies)
                        {
                            double user1Rating = userRatings[i].get(movieId);
                            double user2Rating = userRatings[j].get(movieId);

                            double diff1 = user1Rating-user1Mean;
                            double diff2 = user2Rating-user2Mean;

                            numerator += (diff1*diff2);
                            dev1 += (diff1*diff1);
                            dev2 += (diff2*diff2);
                        }
                    }
                    dev1 = Math.sqrt(dev1);
                    dev2 = Math.sqrt(dev2);

                    denominator = dev1*dev2;
                    if(denominator == 0)
                    {
                        coeff = 0;
                    }
                    else
                    {
                        coeff = (numerator/denominator);
                    }

                    if(commonMovies.isEmpty())
                    {
                        coeffs[i][j] = 0.0;
                        coeffs[j][i] = 0.0;
                    }

                    coeffs[i][j] = coeff;
                    coeffs[j][i] = coeff;
                }
                else
                {
                    coeffs[i][j] = 99999.0;
                }
            }
        }

        File f = new File("src/main/resources/ml-100k/pearsonCoeffs.csv");
        f.createNewFile();
        FileWriter writer = new FileWriter(f);

        writer.write("userId");
        for(int i=1;i<=numUsers;i++)
        {
            writer.append(", " + i);
        }
        writer.append("\n");
        for(int i=1;i<=numUsers;i++)
        {
            writer.append(String.valueOf(i));
            for(int j=1;j<=numUsers;j++)
            {
                writer.append(", " + coeffs[i][j]);
            }
            writer.append("\n");
        }
        writer.close();
    }

    public static Double[] getUserMeans(SparkSession spark, int numUsers)
    {
        Dataset<Row> userMeanData = spark.read().option("header", true)
                .option("inferSchema", true).csv("src/main/resources/ml-100k/userMean.csv");

        userMeans = new Double[numUsers+1];
        for(int i=0;i<=numUsers;i++)
        {
            userMeans[i] = 1.0;
        }

        List<Row> rows = userMeanData.javaRDD().collect();

        for(Row row: rows)
        {
            int userId = row.getInt(0);
            double mean = row.getDouble(1);
            userMeans[userId] = mean;
        }
        return userMeans;
    }

    public static HashMap[] getUserRatings(Dataset<Row> dataset, int numUsers)
    {
        HashMap<Long, Double>[] userRatings = new HashMap[numUsers+1];
        for(int i=1;i<=numUsers;i++)
        {
            userRatings[i] = new HashMap<>();
        }

        JavaRDD<Row> ratingsRDD = dataset.javaRDD();
        JavaPairRDD<Integer, Tuple2<Long,Double>> rdd = ratingsRDD.mapToPair((Row row) -> {
            int userId = row.getInt(0);
            long movieId = Long.parseLong(String.valueOf(row.getInt(1)));
            double rating = row.getDouble(2);
            return new Tuple2<>(userId, new Tuple2<>(movieId, rating));
        });

        List<Tuple2<Integer, Tuple2<Long, Double>>> list = rdd.collect();
        for(Tuple2<Integer, Tuple2<Long, Double>> elem: list)
        {
            int userId = elem._1;
            long movieId = elem._2._1;
            double rating = elem._2._2;
            userRatings[userId].put(movieId,rating);
        }
        return userRatings;
    }

    public static void createUserMeanCSV(Dataset<Row> dataset, SparkSession spark) throws IOException {
        dataset.createOrReplaceTempView("ratings");
        dataset = spark.sql("select userId, avg(rating) from ratings group by userId");

        dataset.createOrReplaceTempView("userMean");
        dataset = spark.sql("select * from userMean order by userId");

        File userMeanFile = new File("src/main/resources/ml-100k/userMean.csv");
        if(userMeanFile.exists())
        {
            userMeanFile.delete();
        }

        File file = new File("src/main/resources/ml-100k/userMeanDir");
        if (file.delete()) {
            System.out.println("Deleted the file: " + file.getName());
        } else {
            System.out.println("Failed to delete the file.");
        }

        String directoryPath = "src/main/resources/ml-100k/userMeanDir";
        String destinationFolder = "src/main/resources/ml-100k";
        File dir = new File(directoryPath);
        File[] filesInDirectory = dir.listFiles();

        if(filesInDirectory != null)
        {
            for(File f : filesInDirectory){
                f.delete();
            }
            dir.delete();
        }

        dataset.coalesce(1).write().format("csv").save("src/main/resources/ml-100k/userMeanDir");

        dir = new File(directoryPath);
        filesInDirectory = dir.listFiles();
        for(File f : filesInDirectory){
            String filePath = f.getAbsolutePath();
            String fileExtenstion = filePath.substring(filePath.lastIndexOf(".") + 1,filePath.length());
            if("csv".equals(fileExtenstion)){
                Files.move(Paths.get(filePath), Paths.get(destinationFolder+"/userMean.csv"));
            }
        }

    }
}
