package examples;

import java.nio.file.Files;
import java.io.File;
import java.util.Arrays;

import shap4j.TreeExplainer;

public class ExampleApp {
    public static void main(String[] args) throws Exception {
        //"/home/stats/shap4jSampleCall/src/main/java/examples/boston.shap4j"

        String fileName = args[0];
        byte[] data = Files.readAllBytes(new File(fileName).toPath());
        TreeExplainer explainer = new TreeExplainer(data);
        double[] x = {
                6.320e-03, 1.800e+01, 2.310e+00, 0.000e+00, 5.380e-01, 6.575e+00,
                6.520e+01, 4.090e+00, 1.000e+00, 2.960e+02, 1.530e+01, 3.969e+02,
                4.980e+00
        };
        double[] shapValues = explainer.shapValues(x, false);

        System.out.println("SHAP values: " + Arrays.toString(shapValues));
    }
}