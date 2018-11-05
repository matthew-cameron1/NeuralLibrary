package io.matthd.neurallibrary.network;

import java.util.Arrays;

public class Network {

    private double[][][] weights;
    private double[][] errors;
    private final int[] networkSizes;
    private double[][] derivatives;
    private double[][] neuronOut;

    public Network(int[] networkSizes) {
        this.networkSizes = networkSizes;
        this.weights = new double[networkSizes.length][][];
        this.errors = new double[networkSizes.length][];
        this.derivatives = new double[networkSizes.length][];
        this.neuronOut = new double[networkSizes.length][];
        for (int layer = 0; layer < networkSizes.length; layer++) {

            this.errors[layer] = new double[networkSizes[layer]];
            this.derivatives[layer] = new double[networkSizes[layer]];
            this.neuronOut[layer] = new double[networkSizes[layer]];

            if (layer > 0) {
                this.weights[layer] = new double[networkSizes[layer]][networkSizes[layer - 1]];
                for (int neuron = 0; neuron < networkSizes[layer]; neuron++) {

                    for (int lastNeuron = 0; lastNeuron < networkSizes[layer - 1]; lastNeuron++) {
                        weights[layer][neuron][lastNeuron] = Math.random();
                    }
                }
            }
        }
    }

    public void train(double[] inputs, double[] targets, double learningRate) {
        guess(inputs);
        backprop(targets);
        updateWeights(learningRate);
    }

    private void backprop(double[] target) {
        for (int neuron = 0; neuron < networkSizes[networkSizes.length - 1]; neuron++) {
            errors[networkSizes.length - 1][neuron] = (neuronOut[networkSizes.length - 1][neuron] - target[neuron]) * derivatives[networkSizes.length - 1][neuron];
        }

        for (int layer = networkSizes.length - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < networkSizes[layer]; neuron++) {
                double weightedSum = 0;
                for (int nextNeuron = 0; nextNeuron < networkSizes[layer + 1]; nextNeuron++) {
                    weightedSum += weights[layer + 1][nextNeuron][neuron] * errors[layer + 1][nextNeuron];
                }
                this.errors[layer][neuron] = weightedSum * derivatives[layer][neuron];
            }
        }
    }

    private void updateWeights(double learningRate) {
        for (int layer = 1; layer < networkSizes.length; layer++) {
            for (int neuron = 0; neuron < networkSizes[layer]; neuron++) {
                double delta = -learningRate * errors[layer][neuron];

                for (int lastNeuron = 0; lastNeuron < networkSizes[layer - 1]; lastNeuron++) {
                    weights[layer][neuron][lastNeuron] += delta * neuronOut[layer - 1][lastNeuron];
                }
            }
        }
    }

    public double[] guess(double[] inputs) {
        this.neuronOut[0] = inputs;
        for (int layer = 1; layer < networkSizes.length; layer++) {
            for (int neuron = 0; neuron < networkSizes[layer]; neuron++) {
                double sum = 0;
                for (int lastNeuron = 0; lastNeuron < networkSizes[layer - 1]; lastNeuron++) {
                    sum += neuronOut[layer - 1][lastNeuron] * weights[layer][neuron][lastNeuron];
                }
                neuronOut[layer][neuron] = sigmoid(sum);
                derivatives[layer][neuron] = sigmoidDeriv(neuronOut[layer][neuron]);
            }
        }
        return neuronOut[networkSizes.length - 1];
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private double sigmoidDeriv(double x) {
        return 1 * (1 - x);
    }

    public static void main(String[] args) {
        Network network = new Network(new int[]{2, 3, 1});

        double[] input = new double[]{1, 1};
        double[] output = new double[]{0};

        for (int epochs = 0; epochs < 1000; epochs++) {
            network.train(input, output, 0.3);
            //Train our network to the new data
        }

        //Lets see if we get the correct output after our training data
        System.out.println("The network output for input: " + Arrays.toString(input) + " is:");
        System.out.println(Arrays.toString(network.guess(input)));
    }
}
