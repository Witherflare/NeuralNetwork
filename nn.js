import { Matrix } from './matrix.js';

export default class NeuralNetwork {
  constructor(numI, numH, numO, activation) { // Initialize our neural network
    this.input_nodes = numI; // Number of input nodes in the network
    this.hidden_nodes = numH; // Number of hidden nodes (also called processors or neurons) in the network
    this.output_nodes = numO; // Number of output nodes in the network

    this.activation = activation; // The activation function to use

    this.a = this.a.bind(this);
    this.a_derivative = this.a_derivative.bind(this);

    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes); // The weights between input and hidden layers (i-h)
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes); // The weights between hidden and output layers (h-o)
    this.weights_ih.randomize(); // Initialize the i-h weights by giving them a random value from -1 to 1
    this.weights_ho.randomize(); // Initialize the h-o weights by giving them a random value from -1 to 1

    this.bias_h = new Matrix(this.hidden_nodes, 1); // The bias for the hidden layer (matrix)
    this.bias_o = new Matrix(this.output_nodes, 1); // The bias for the output layer (matrix)
    this.bias_h.randomize(); // Initialize the hidden layer bias by giving it a random value from -1 to 1
    this.bias_o.randomize(); // Initialize the output layer bias by giving it a random value from -1 to 1
  
    this.learning_rate = 0.1; // The learning rate for the network
  }

  a(x) {
    switch (this.activation) {
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x)); // Sigmoid function (https://en.wikipedia.org/wiki/Sigmoid_function)
      case 'tanh':
        return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)); // Hyperbolic tangent function (https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent)
      case 'relu':
        return x > 0 ? x : 0; // Rectified linear unit function (https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
    }
  }

  a_derivative(x) {
    switch (this.activation) {
      case 'sigmoid':
        return this.a(x) * (1 - this.a(x)); // Derivative of the sigmoid function (s'(x) = s(x) * (1 - s(x)))
      case 'tanh':
        return 1 - this.a(x) * this.a(x); // Derivative of the hyperbolic tangent function (tanh'(x) = 1 - tanh(x)^2)
      case 'relu':
        return x > 0 ? 1 : 0; // Derivative of the rectified linear unit function (relu'(x) = 1 if x > 0, 0 otherwise)
    }
  }

  setWeights(data) {
    // set parameters based on json
    this.weights_ih = Matrix.fromJSON(data.weights_ih);
    this.weights_ho = Matrix.fromJSON(data.weights_ho);
    this.bias_h = Matrix.fromJSON(data.bias_h);
    this.bias_o = Matrix.fromJSON(data.bias_o);
    this.input_nodes = data.input_nodes;
    this.hidden_nodes = data.hidden_nodes;
    this.output_nodes = data.output_nodes;
  }

  getWeights() {
    // return data
    return {
      weights_ih: this.weights_ih.toJSON(),
      weights_ho: this.weights_ho.toJSON(),
      bias_h: this.bias_h.toJSON(),
      bias_o: this.bias_o.toJSON(),
      input_nodes: this.input_nodes,
      hidden_nodes: this.hidden_nodes,
      output_nodes: this.output_nodes
    }
  }

  run(raw_input) { // Our feed-forward algorithm, for inputting data and receiving an output
    let input = Matrix.fromArray(raw_input); // Convert the input array into a matrix
    let hidden = Matrix.multiply(this.weights_ih, input); // Get the weighted sum for the hidden layer (matrix)
    hidden.add(this.bias_h); // Add the bias to the weighted sum (matrix)
    hidden.map(this.a); // Apply the activation function to the weighted sum
    let output = Matrix.multiply(this.weights_ho, hidden); // Get the weighted sum for the output layer (matrix)
    output.add(this.bias_o); // Add the bias to the weighted sum (matrix)
    output.map(this.a) // Apply the activation function to the weighted sum (matrix)
    return output.toArray(); // Convert the matrix into an array
  }

  train(raw_inputs, expecteds) { // Train the neural network with an input array and an expected output array
    // This is just the "run" function
    let input = Matrix.fromArray(raw_inputs); // Convert the input array into a matrix
    let hidden = Matrix.multiply(this.weights_ih, input); // Get the weighted sum for the hidden layer (matrix)
    hidden.add(this.bias_h); // Add the bias to the weighted sum (matrix)
    let hidden_gradients = Matrix.map(hidden, this.a_derivative); // Map the outputs through the derivative of the sigmoid function (we'll use this later)
    hidden.map(this.a); // Apply the activation function to the weighted sum (matrix)
    let outputs = Matrix.multiply(this.weights_ho, hidden); // Get the weighted sum for the output layer (matrix)
    outputs.add(this.bias_o); // Add the bias to the weighted sum (matrix)
    let gradients = Matrix.map(outputs, this.a_derivative); // Map the outputs through the derivative of the sigmoid function (we'll use this later)
    outputs.map(this.a) // Apply the activation function to the weighted sum (matrix)

    expecteds = Matrix.fromArray(expecteds); // Convert the expected array into a matrix
    let output_errors = Matrix.subtract(expecteds, outputs) // Calculate how far off the output is from the expected output (error = expected - output)
    
    // Now, we need to find the errors for the hidden layer.
    // This can be done by multiplying the errors by the transpose (flip) of the weights_ho matrix.
    // This will give us the errors for the hidden layer.
    let hidden_errors = Matrix.multiply(Matrix.transpose(this.weights_ho), output_errors); // Multiply the transpose of the weights_ho matrix by the errors (matrix)
    
    // lr * E * s'(h) * x (here, we're not multiplying by x, as we're only trying to find the gradient h-o)
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    // we've done lr * E * s'(h) (calculated the gradient), now we need to do matrix multiplication with the transpose of hidden (get the deltas)
    let hidden_T = Matrix.transpose(hidden); // getting the transpose
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_T); // Multiply the transpose of the hidden layer by the gradient (matrix)

    // Repeat the process for i-h weights
    hidden_gradients.multiply(hidden_errors);
    hidden_gradients.multiply(this.learning_rate);

    // Now, we calculate the deltas for the i-h weights
    let inputs_T = Matrix.transpose(input);
    let weight_ih_deltas = Matrix.multiply(hidden_gradients, inputs_T);

    // Updating the weights!
    this.weights_ho.add(weight_ho_deltas);
    this.weights_ih.add(weight_ih_deltas);
    // Updating the biases by their deltas (which is just the gradients)
    this.bias_o.add(gradients);
    this.bias_h.add(hidden_gradients);
  }
}

let training_data = [
  {
    inputs: [0, 0, 1],
    output: [0]
  },
  {
    inputs: [0, 1, 1],
    output: [1]
  },
  {
    inputs: [1, 0, 1],
    output: [1]
  },
  {
    inputs: [0, 1, 0],
    output: [1]
  },
  {
    inputs: [1, 1, 1],
    output: [0]
  },
  {
    inputs: [0, 0, 0],
    output: [0]
  },
]

let network = new NeuralNetwork(3, 3, 1, 'sigmoid');

for (var i = 0; i < 100000; i++) {
  let data = training_data[Math.floor(Math.random() * training_data.length)];
  network.train(data.inputs, data.output);
}

console.log(network.run([1, 0, 0]));