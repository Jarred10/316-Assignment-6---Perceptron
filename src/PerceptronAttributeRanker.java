import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Scanner;


public class PerceptronAttributeRanker {

	public static void main(String[] args) throws FileNotFoundException {

		HashSet<Integer> ignored = new HashSet<Integer>();

		//bias input, always -1
		final int BIAS = -1;
		
		//alpha, user specified, defines the learning rate
		final double alpha = Double.valueOf(args[1]);


		Scanner s = new Scanner(new File(args[0]));
		
		//find headers
		String[] headers = s.nextLine().split(",");
		
		int lineCount = 0;
		
		while(s.hasNextLine()){
			s.nextLine();
			lineCount++;
		}

		while(ignored.size() < 100){

			//array of weights, with same indexes as columns in data
			double[] weights = new double[101];
			double[] averageAbsoluteWeights = new double[101];

			//total classification errors counter
			int correctClassifications = 0;
			
			//start reading data again
			s = new Scanner(new File(args[0]));
			//skip headers
			s.nextLine();
			
			//while there is data left to work with
			while(s.hasNextLine()){
				double inputSum = 0;
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
					inputSum += (Integer.valueOf(values[i]) * weights[i]);
				}
				inputSum +=  BIAS * weights[100];

				double output = sigmoid(inputSum);

				int expectedValue = (int) Math.round(output);

				int actualValue = values[100];

				if(expectedValue == actualValue) correctClassifications++;

				double error = Integer.valueOf(actualValue) - output;

				double derivative = output * (1 - output);

				for(int i = 0; i < 100; i++){
					weights[i] = weights[i] + (alpha * (error * derivative * values[i]));

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


			System.out.println("Highest impact input: " + headers[index] + ", weight: " + averageAbsoluteWeights[index] / lineCount + ". Correct classifications: " + correctClassifications);

			ignored.add(index);
			
			s.close();
		}

	}

	public static double sigmoid(double value){
		return 1/(1 + Math.pow(Math.E, -value));
	}
}

