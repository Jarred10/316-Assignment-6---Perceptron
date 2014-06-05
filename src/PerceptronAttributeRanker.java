import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Scanner;


public class PerceptronAttributeRanker {

	public static void main(String[] args) throws FileNotFoundException {

		HashSet<Integer> ignored = new HashSet<Integer>();

		//bias input, always -1
		final double BIAS = -1;
		//alpha, user specified, defines the learning rate
		final double alpha = Double.valueOf(args[1]);

		//find headers

		while(ignored.size() < 100){

			//array of weights, with same indexes as columns in data
			double[] weights = new double[101];
			double[] averageAbsoluteWeights = new double[101];

			//total classification errors counter
			int totalCorrectClassifications = 0;

			int iterations = 0;
			
			Scanner s = new Scanner(new File(args[0]));
			
			String[] headers = s.nextLine().split(",");
			
			//while there is data left to work with
			while(s.hasNextLine()){
				iterations++;
				double inputFunctionValue = 0;
				String data = s.nextLine();
				int[] values = new int[101];
				for(int i = 0; i < values.length; i++){
					if(!ignored.contains(i)){
						values[i] = data.charAt(i*2) - '0';
					}
					else{
						values[i] = 0;
					}
				}

				for(int i = 0; i < 100; i++){
					inputFunctionValue += (Integer.valueOf(values[i]) * weights[i]);
				}
				inputFunctionValue += weights[100] * BIAS;

				inputFunctionValue = sigmoid(inputFunctionValue);

				int expectedValue = (int) Math.round(inputFunctionValue);

				int actualValue = Integer.valueOf(values[100]);

				if(expectedValue == actualValue) totalCorrectClassifications++;

				double error = Integer.valueOf(actualValue) - inputFunctionValue;

				double derivative = inputFunctionValue * (1 - inputFunctionValue);

				for(int i = 0; i < 100; i++){
					weights[i] = weights[i] + (alpha * (error * derivative * Integer.valueOf(values[i])));

					averageAbsoluteWeights[i] += Math.abs(weights[i]);
				}

				weights[100] = weights[100] + (alpha * (error * derivative * BIAS));

				//System.out.println("Expected: " + expectedValue + ". Actual: " + actualValue + ". " + "Error: " + error);

			}

			double highest = 0;
			int index = 0;

			for(int i = 0; i < averageAbsoluteWeights.length; i++){
				if(averageAbsoluteWeights[i] > highest) {
					highest = averageAbsoluteWeights[i];
					index = i;
				}
			}


			System.out.println("Highest impact input: " + headers[index] + ", weight: " + averageAbsoluteWeights[index] / iterations + ". Correct classifications: " + totalCorrectClassifications);

			ignored.add(index);
			
			s.close();
		}

	}

	public static double sigmoid(double value){
		return 1/(1 + Math.pow(Math.E, -value));
	}
}

