import { Matrix } from './matrix.mjs';

function sigmoid (x) {
  return 1 / (1 + Math.exp(-x)); // Sigmoid function (https://en.wikipedia.org/wiki/Sigmoid_function)
}

class NeuralNetwork {
  constructor(numI, numH, numO) { // Initialize our neural network
    this.input_nodes = numI; // Number of input nodes in the network
    this.hidden_nodes = numH; // Number of hidden nodes (also called processors or neurons) in the network
    this.output_nodes = numO; // Number of output nodes in the network

    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes); // The weights between input and hidden layers (matrix)
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes); // The weights between hidden and output layers (matrix)
    this.weights_ih.randomize(); // Initialize the i-h weights by giving them a random value from -1 to 1
    this.weights_ho.randomize(); // Initialize the h-o weights by giving them a random value from -1 to 1

    this.bias_h = new Matrix(this.hidden_nodes, 1); // The bias for the hidden layer (matrix)
    this.bias_o = new Matrix(this.output_nodes, 1); // The bias for the output layer (matrix)
    this.bias_h.randomize(); // Initialize the hidden layer bias by giving it a random value from -1 to 1
    this.bias_o.randomize(); // Initialize the output layer bias by giving it a random value from -1 to 1
  }

  run(raw_input) { // Our feed-forward algorithm, for inputting data and receiving an output
    let input = Matrix.fromArray(raw_input); // Convert the input array into a matrix
    let hidden = Matrix.multiply(this.weights_ih, input); // Get the weighted sum for the hidden layer (matrix)
    hidden.add(this.bias_h); // Add the bias to the weighted sum (matrix)
    hidden.map(sigmoid); // Apply the sigmoid function to the weighted sum (matrix)
    let output = Matrix.multiply(this.weights_ho, hidden); // Get the weighted sum for the output layer (matrix)
    output.add(this.bias_o); // Add the bias to the weighted sum (matrix)
    output.map(sigmoid) // Apply the sigmoid function to the weighted sum (matrix)
    return output.toArray(); // Convert the matrix into an array
  }

  train(inputs, targets) { // Train the neural network with an input array and an expected output array
    
  }
}

let nn = new NeuralNetwork(2, 2, 1); // Create a neural network with 2 input nodes, 2 hidden nodes, and 1 output node
console.log(nn.feedForward([1, 0]).join(", ")); // Feed data into the network and recieve an output