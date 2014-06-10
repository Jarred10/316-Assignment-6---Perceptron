import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


public class AdaptiveLearningRatePerceptron {
	
	//bias input, always -1
	final static int BIAS = -1;

	static double[] alphaWeights = new double[101];
	static double[] gammaWeights = new double[101];
	static double[] inverseGammaWeights = new double[101];

	public static void main(String[] args) throws IOException {
		
		//first arg is filename
		BufferedReader rd = new BufferedReader(new FileReader(args[0]));
		
		//skip headers
		rd.readLine();
		
		//second arg is learning rate alpha
		double _alpha = Double.valueOf(args[1]);
		
		//third arg is batch size N
		int batchSize = Integer.valueOf(args[2]);
		
		//fourth argument is multiplier gamma
		double _gamma = Double.valueOf(args[3]);
		
		//first learning rate alpha
		double alpha = _alpha;
		int alphaErrors = 0;
		
		//second learning rate gamma * alpha
		double gamma = _gamma * _alpha;
		int gammaErrors = 0;
		
		//third learning rate 1/gamma * alpha
		double inverseGamma = (1 / _gamma) * _alpha;
		int inverseGammaErrors = 0;
		
		double alphaNew = 0;
		
		int totalErrors = 0;
		
		String line;
		int linesRead = 0;
		while((line = rd.readLine()) != null){
			if(linesRead == batchSize){
				if(alphaErrors <= gammaErrors && alphaErrors <= inverseGammaErrors){
					alphaNew = alpha;
					totalErrors += alphaErrors;
				}
				else if(gammaErrors <= alphaErrors && gammaErrors <= inverseGammaErrors){
					alphaNew = gamma;
					totalErrors += gammaErrors;
				}
				else {
					alphaNew = inverseGamma;
					totalErrors += inverseGammaErrors;
				}
				
				alphaErrors = 0; gammaErrors = 0; inverseGammaErrors = 0;
				
				alpha = alphaNew;

				System.out.println("New alpha: " + alpha);
				gamma = _gamma * alpha;
				inverseGamma = (1 / _gamma) * alphaNew;
				linesRead = 0;
			}
			
			linesRead++;
			
			alphaErrors += train(alpha, line, alphaWeights);
			gammaErrors += train(gamma, line, gammaWeights);
			inverseGammaErrors += train(inverseGamma, line, inverseGammaWeights);			
		}
		
		System.out.println("Total errors: " + totalErrors);
		
		
	}
	
	public static int train(double alpha, String data, double[] weights){
		int classificationErrors = 0;
		
		double inputSum = 0;
		int[] values = new int[100];
		for(int i = 0; i < values.length; i++){
			//if we havent removed the input, get the value, else leave value as 0
			values[i] = data.charAt(i*2) - '0';
		}
		
		//set the actual value of polarity
		int actualValue = data.charAt(data.length() - 1) - '0';
		
		//for all the values except bias, sum the value * weight
		for(int i = 0; i < values.length; i++) inputSum += values[i] * weights[i];
		
		inputSum += BIAS * weights[100];

		//sigmoid function applied to sum, 1/(1 + e^-x)
		double output =  1/(1 + Math.pow(Math.E, -inputSum));

		//round sigmoid value by casting to int after adding 0.5 (floor)
		int expectedValue = (int) (output + 0.5);

		if(expectedValue != actualValue) classificationErrors++;

		double error = actualValue - output;

		double derivative = output * (1 - output);

		//update the input weights and sum their absolute weights
		for(int i = 0; i < 100; i++){
			weights[i] = weights[i] + (alpha * (error * derivative * values[i]));
		}

		//update the bias weight
		weights[100] = weights[100] + (alpha * (error * derivative * BIAS));
		
		
		return classificationErrors;
	}

}
