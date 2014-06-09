import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.Scanner;

public class PerceptronAttributeRanker {

	public static void main(String[] args) throws FileNotFoundException {

		//list of lines of the data, stored so we don't have to reread
		ArrayList<String> data = new ArrayList<String>();

		//bias input, always -1
		final int BIAS = -1;
		
		//alpha, user specified, defines the learning rate
		final double alpha = Double.valueOf(args[1]);

		//read in file
		Scanner s = new Scanner(new File(args[0]));
		
		//store headers
		String[] headers = s.nextLine().split(",");
		
		//construct list of lines in data
		while(s.hasNextLine()) data.add(s.nextLine());

		s.close();
		
		//linked hash set of all indexes of inputs to use
		LinkedHashSet<Integer> currentInputs = new LinkedHashSet<Integer>();
		
		//add all indexes of input to initial list of current inputs
		for (int i = 0; i < headers.length - 1; i++){
			currentInputs.add(i);
		}
		
		//while their are inputs left to test and remove
		while(currentInputs.size() > 0){

			//array of weights, with same indexes as columns in data
			double[] weights = new double[101];
			
			//array of absolute weights, indexes as same as weights. excludes bias weight
			double[] absoluteWeights = new double[100];

			//total classification errors counter
			int correctClassifications = 0;
			
			//while there is data left to work with
			for(String line : data){
				double inputSum = 0;
				int[] values = new int[100];
				for(int i = 0; i <= values.length; i++){
					//if we havent removed the input, get the value, else leave value as 0
					if(currentInputs.contains(i)) values[i] = line.charAt(i*2) - '0';
				}
				
				//set the actual value of polarity
				int actualValue = line.charAt((headers.length - 1) * 2) - '0';
				
				//for all the values except bias, sum the value * weight
				for(int i = 0; i < values.length; i++) inputSum += values[i] * weights[i];
				
				inputSum += BIAS * weights[100];

				//sigmoid function applied to sum, 1/(1 + e^-x)
				double output =  1/(1 + Math.pow(Math.E, -inputSum));

				//round sigmoid value by casting to int after adding 0.5 (floor)
				int expectedValue = (int) (output + 0.5);

				if(expectedValue == actualValue) correctClassifications++;

				double error = actualValue - output;

				double derivative = output * (1 - output);

				//update the input weights and sum their absolute weights
				for(int i = 0; i < 100; i++){
					weights[i] = weights[i] + (alpha * (error * derivative * values[i]));
					absoluteWeights[i] += Math.abs(weights[i]);
				}

				//update the bias weight
				weights[100] = weights[100] + (alpha * (error * derivative * BIAS));
				
			}

			//set values to safe, non-null and impossible defaults
			double largest = Double.NEGATIVE_INFINITY;
			int index = -1;

			//for all current inputs, find the one with largest absolute weight
			for(int input : currentInputs){
				if(absoluteWeights[input] > largest) {
					largest = absoluteWeights[input];
					index = input;
				}
			}

			System.out.println("Highest impact input: " + headers[index] + ", weight: " + absoluteWeights[index] / data.size() + ". Correct classifications: " + correctClassifications);

			//remove the most influencial input
			currentInputs.remove(index);
			
		}

	}
}

