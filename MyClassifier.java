import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MyClassifier {

    public static void main(String[] args) {
        // Parsing cmd line args
        if(args.length != 3) {
            throw new IllegalArgumentException("Invalid number of arguments");
        }
        String trainingPath = args[0];
        String testPath = args[1];
        String algo = args[2];

        // Reading files
        List<Example> trainingData = new ArrayList<>();
        List<Example> testData = new ArrayList<>();
        try{
            Stream<String> lines = Files.lines(Paths.get(trainingPath));
            List<String> temp = lines.collect(Collectors.toList());
            for(String line : temp) {
                List<String> vals = Arrays.asList(line.split(","));
                List<Double> valsList = new ArrayList<>();
                for(int i = 0; i < vals.size() - 1; i++) {
                    valsList.add(Double.parseDouble(vals.get(i)));
                }
                trainingData.add(new Example(valsList, vals.get(vals.size() - 1)));
            }

            lines = Files.lines(Paths.get(testPath));
            temp = lines.collect(Collectors.toList());
            for(String line : temp) {
                List<String> vals = Arrays.asList(line.split(","));
                List<Double> valsList = new ArrayList<>();
                for(String val : vals) {
                    valsList.add(Double.parseDouble(val));
                }
                testData.add(new Example(valsList, null));
            }

        } catch (IOException e) {
            e.printStackTrace();
            System.exit(-1);
        }

        List<String> results = classify(algo, trainingData, testData);
        //System.out.println(evaluate(algo, trainingData));

        if(results != null) {
            for(String result : results) {
                System.out.println(result);
            }
        }
    }

    public static List<String> classify(String algo, List<Example> trainingData, List<Example> testData) {
        List<String> results = null;
        // Selecting an algorithm
        if(algo.equals("NB")) {
            results = naiveBayes(trainingData, testData);
        }
        if(algo.matches("^(\\d+[N]{2})$")) {
            int k = Integer.parseInt(algo.substring(0, algo.length()-2));
            results = kNearest(trainingData, testData, k);
        }
        return results;
    }

    public static double evaluate(String algo, List<Example> trainingData) {
        List<List<Example>> folds = nFoldStratCrossValidation(10, trainingData);
        double accuracy = 0.0;
        for(int i = 0; i < folds.size(); i++) {
            List<Example> trainingFolds = new ArrayList<>();
            List<Example> testFold = folds.get(i);
            for(int j = 0; j < folds.size(); j++) {
                if(i == j) {
                    continue;
                }
                trainingFolds.addAll(folds.get(j));
            }
            List<String> results = classify(algo, trainingFolds, testFold);
            double nCorrect = 0;
            for(int j = 0; j < results.size(); j++) {
                if(results.get(j).equals(testFold.get(j).eClass)) {
                    nCorrect++;
                }
            }
            accuracy += nCorrect/results.size();
        }
        generateFoldsFile(10, folds);
        return accuracy/folds.size();
    }

    public static List<String> kNearest(List<Example> trainingData, List<Example> testData, int k) {
        List<String> results = new ArrayList<>();
        for(Example example: testData) {
            for(Example entry: trainingData) {
                double distance = 0;
                for(int i = 0; i < entry.vals.size(); i++) {
                    distance += Math.pow(entry.vals.get(i) - example.vals.get(i), 2);
                }
                distance = Math.pow(distance, 0.5);
                entry.distance = distance;
            }
            trainingData.sort(Comparator.comparingDouble(a -> a.distance));
            int yes = 0;
            int no = 0;
            for(int i = 0; i < k; i++) {
                if(trainingData.get(i).eClass.equals("yes")) {
                    yes++;
                } else {
                    no++;
                }
            }

            if(yes >= no) {
                results.add("yes");
            } else{
                results.add("no");
            }
        }
        return results;
    }

    public static List<String> naiveBayes(List<Example> trainingData, List<Example> testData) {
        List<String> results = new ArrayList<>();
        List<Double> yesMean = new ArrayList<>();
        List<Double> yesSD = new ArrayList<>();
        List<Double> noMean = new ArrayList<>();
        List<Double> noSD = new ArrayList<>();
        // Calculate Mean
        int yes = 0;
        int no = 0;
        for(Example example : trainingData) {
            if(example.eClass.equals("yes")) {
                yes++;
            } else {
                no++;
            }
        }
        for(int i = 0; i < trainingData.get(0).vals.size(); i++) {
            yesMean.add(0.0);
            noMean.add(0.0);
            for (Example trainingDatum : trainingData) {
                if (trainingDatum.eClass.equals("yes")) {
                    yesMean.set(i, yesMean.get(i) + trainingDatum.vals.get(i));
                } else {
                    noMean.set(i, noMean.get(i) + trainingDatum.vals.get(i));
                }
            }
        }
        for(int i = 0; i < trainingData.get(0).vals.size(); i++) {
            yesMean.set(i, yesMean.get(i) / yes);
            noMean.set(i, noMean.get(i) / no);
        }
        // Calculate S.D
        for(int i = 0; i < trainingData.get(0).vals.size(); i++) {
            yesSD.add(0.0);
            noSD.add(0.0);
            for (Example example : trainingData) {
                if (example.eClass.equals("yes")) {
                    yesSD.set(i, yesSD.get(i) + Math.pow(example.vals.get(i) - yesMean.get(i), 2));
                } else {
                    noSD.set(i, noSD.get(i) + Math.pow(example.vals.get(i) - noMean.get(i), 2));
                }
            }
        }
        for(int i = 0; i < trainingData.get(0).vals.size(); i++) {
            yesSD.set(i, Math.pow(yesSD.get(i) / (yes- 1), 0.5));
            noSD.set(i, Math.pow(noSD.get(i) / (no - 1), 0.5));
        }
        // Classify test data
        for(Example example : testData) {
            double pYes = (double) yes /(yes + no);
            double pNo = (double) no /(yes + no);
            for(int i = 0; i < example.vals.size(); i++) {
                double ySD = yesSD.get(i);
                double nSD = noSD.get(i);

                pYes = pYes * (1/(ySD* Math.pow(2*Math.PI, 0.5))) * Math.pow( Math.E,
                        -(Math.pow(example.vals.get(i) - yesMean.get(i), 2)/(2*Math.pow(ySD,2))));
                pNo = pNo * (1/(nSD* Math.pow(2*Math.PI, 0.5))) *Math.pow( Math.E,
                        -(Math.pow(example.vals.get(i) - noMean.get(i), 2)/(2*Math.pow(nSD,2))));
            }
            if(pYes >= pNo) {
                results.add("yes");
            } else {
                results.add("no");
            }
        }
        return  results;
    }

    public static class Example {
        List<Double> vals;
        String eClass;
        double distance = 0.0;
        public Example(List<Double> vals, String eClass) {
            this.vals = vals;
            this.eClass = eClass;
        }

    }

    public static List<List<Example>> nFoldStratCrossValidation(int n, List<Example> trainingData) {
        List<List<Example>> folds = new ArrayList<>();
        for(int i = 0; i < n; i++) {
            folds.add(new ArrayList<>());
        }
        int count = 0;
        for(int i = 0; i < trainingData.size(); i++) {
            if(trainingData.get(i).eClass.equals("yes")) {
                folds.get(count % n).add(trainingData.get(i));
                count++;
            }
        }
        for(int i = 0; i < trainingData.size(); i++) {
            if(trainingData.get(i).eClass.equals("no")) {
                folds.get(count % n).add(trainingData.get(i));
                count++;
            }
        }
        return folds;
    }

    public static void generateFoldsFile(int n, List<List<Example>> folds) {
        // Generate the folds file
        try{
            PrintWriter writer = new PrintWriter("pima-folds.csv");
            for(int i = 0 ; i < n; i++) {
                writer.println("fold" + (i+1));
                for(Example example : folds.get(i)) {
                    for(Double val : example.vals) {
                        writer.print(val + ",");
                    }
                    writer.println(example.eClass);
                }
                if(i != n - 1) {
                    writer.println("");
                }
            }
            writer.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

}
