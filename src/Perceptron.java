import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;


public class Perceptron {

	public static void main(String[] args) throws FileNotFoundException {
		
		//scanner that reads twitter sentiment data
		Scanner s = new Scanner(new File(args[0]));
		
		//array of weights, with same indexes as columns in data
		double[] weights = new double[101];
		
		
		//bias input, always -1
		final double BIAS = -1;
		//alpha, user specified, defines the learning rate
		final double alpha = Double.valueOf(args[1]);
		
		//find headers
		String[] headers = s.nextLine().split(",");
		
		//total classification errors counter
		int totalClassificationErrors = 0;
		
		//while there is data left to work with
		while(s.hasNextLine()){
			double inputFunctionValue = 0;
			String data = s.nextLine();
			int[] values = new int[101];
			for(int i = 0; i < values.length; i++){
				values[i] = data.charAt(i*2) - '0';
			}
			
			for(int i = 0; i < 100; i++){
				inputFunctionValue += (Integer.valueOf(values[i]) * weights[i]);
			}
			inputFunctionValue += weights[100] * BIAS;
			
			double activationFunctionValue = sigmoid(inputFunctionValue);
			
			int expectedValue = (int) Math.round(activationFunctionValue);
			
			int actualValue = Integer.valueOf(values[100]);
			
			if(expectedValue != actualValue) totalClassificationErrors++;
			
			double error = Integer.valueOf(actualValue) - activationFunctionValue;
			
			double squaredError = Math.pow(error, 2) * 0.5;
			if(error < 0) squaredError *= -1;
			
			double derived = derivitive(activationFunctionValue);
			
			for(int i = 0; i < 100; i++){
				weights[i] = weights[i] + (alpha * (error * derived * Integer.valueOf(values[i])));
			}
			
			weights[100] = weights[100] + (alpha * (error * derived * BIAS));
			
			//System.out.println("Expected: " + expectedValue + ". Actual: " + actualValue + ". " + "Error: " + error);
			
		}
		
		System.out.println("Total classification errors: " + totalClassificationErrors);
		
		s.close();
	}
	
	public static double sigmoid(double value){
		return 1/(1 + Math.pow(Math.E, -value));
	}
	
	public static double derivitive(double value){
		return value * (1 - value);
	}

}
