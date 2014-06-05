import java.io.File;
import java.io.FileNotFoundException;
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
			
			double inputSum = 0;
			
			String data = s.nextLine();
			
			int[] values = new int[101];
			
			for(int i = 0; i < values.length; i++){
				values[i] = data.charAt(i*2) - '0';
			}
			
			for(int i = 0; i < 100; i++){
				inputSum += (Integer.valueOf(values[i]) * weights[i]);
			}
			
			inputSum += weights[100] * BIAS;
			
			//sigmoid function
			double output = 1/(1 + Math.pow(Math.E, -inputSum));
			
			//rounded output
			int expectedValue = (int) Math.round(output);
			
			//value for polarity actually read in
			int actualValue = Integer.valueOf(values[100]);
			
			//increment error if our expected doesnt match actual
			if(expectedValue != actualValue) totalClassificationErrors++;
			
			//error is equal to actual value minus output value
			double error = actualValue - output;
			
			//find derivative, X * (1 - X)
			double derivative = output * (1 - output);
			
			//update weights
			for(int i = 0; i < 100; i++){
				weights[i] = weights[i] + (alpha * (error * derivative * Integer.valueOf(values[i])));
			}
			
			//update the bias weight
			weights[100] = weights[100] + (alpha * (error * derivative * BIAS));
			
			//System.out.println("Expected: " + expectedValue + ". Actual: " + actualValue + ". " + "Error: " + error);
			
		}
		
		System.out.print("<" + weights[100] + ">");
		for(int i = 0; i < 100; i++){
			System.out.print("<" + weights[i] + ">");
		}
		System.out.println();
		System.out.println("<" + totalClassificationErrors + ">");
		
		s.close();
	}
}
