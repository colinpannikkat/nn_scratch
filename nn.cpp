#include "nn.h"

template <class T>
void print(T x) { std::cout << x << std::endl; }

NN::NN(std::vector<uint>topology, Scalar learning_rate) {
    this->learning_rate = learning_rate;
    this->topology = topology;

    /* topology.size() refers to number of layers */
    for (int i = 0; i < topology.size(); i++) {
        // initialize neuron layers, add bias node to every layer but output
        if (i == topology.size() - 1) // if last node
            layers.push_back(new Layer(topology[i], 0));
        else if (i == 0) // if first node
            layers.push_back(new Layer(topology[i], topology[i + 1]));
        else
            layers.push_back(new Layer(topology[i] + 1, topology[i + 1]));
    }

    return;
}

NN::~NN() {
    for (int i = 0; i < topology.size(); i++) {
        if (layers[i] != nullptr) {
            delete layers[i];
            layers[i] = nullptr;
        }
    }
}

void NN::forward(vec input) {

    /* Set first layer to input */
    layers[0]->values = &input;

    for (int i = 1; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->num_nodes; j++) {  
            layers[i]->values->operator[](j) = dot_product(*layers[i - 1]->values, *layers[i - 1]->weights[j]);
        }
    }
    layers[2]->print_values();

    return;
}

void NN::backprop(Scalar output) {
    
    loss(output);
    updateWeights();

    return;
}

void NN::loss(Scalar output) {

    layers[topology.size() - 1]->gradients->back() = output - layers[topology.size() - 1]->values->back();

    for (int i = layers.size() - 2; i > 0; i--) {
        for (int j = 0; j < layers[i]->num_nodes; j++) {
            for (int k = 0; k < layers[i]->gradients->size(); k++) {
                layers[i]->gradients->operator[](0) = layers[i + 1]->gradients->operator[](0) * layers[i]->weights[j]->operator[](k);
            }
        }
    }
    

    return;
}

void NN::updateWeights() {

    for (int i = 1; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->num_nodes; j++) {  
            for (int k = 0; k < layers[i]->weights[j]->size(); k++) {
                layers[i]->weights[j]->operator[](k) = layers[i]->weights[j]->operator[](k) * learning_rate * activationFunctionDerivative(layers[i]->weights[j+1]->operator[](k)) * 1;
            }
        }
    }

    return;
}

void NN::train(std::vector <vec> input, vec output) {

    /* Training loop */
    for (int i = 0; i < input.size(); i++) {
        print("Propogating forward");
        forward(input[i]);
        print("Backprop");
        backprop(output[i]);
    }

    return;
}

Scalar NN::dot_product(vec a, vec b) {

    Scalar res = 0;
    for (int i = 0; (i != a.size() && i != b.size()); i++) {
        if (i == a.size()) {
            res += b[i];
            continue;
        }
        if (i == b.size()) {
            res += a[i];
            continue;
        }
        res += (a[i] * b[i]);
        // print(res);
    }
    return res;
}


Scalar NN::activationFunction(Scalar x)
{
    return std::tanhf(x);
}
 
Scalar NN::activationFunctionDerivative(Scalar x)
{
    return 1 - std::tanhf(x) * std::tanhf(x);
}

/* Constructor for Layer */
NN::Layer::Layer(int num_nodes, int next_num_nodes) {
    this->num_nodes = num_nodes;
    this->next_num_nodes = next_num_nodes;

    // initialize with random weights
    for (int i = 0; i < num_nodes; i++) {
        weights.push_back(new vec(next_num_nodes, (float)rand() / RAND_MAX));
    }
    weights.push_back(new vec(next_num_nodes, 1));
    gradients = new vec(num_nodes);

    // if not one node/output node, add a bias node
    if (num_nodes - 1 != 0) {
        values = new vec(num_nodes - 1);
        values->push_back((Scalar)1); // bias node
    } else {
        values = new vec(num_nodes);
    }
}
/* Destructor for Layer */
NN::Layer::~Layer() {
    for (int i = 0; i < weights.size(); i++) {
        if (weights[i] != nullptr) {
            delete weights[i];
            weights[i] = nullptr;
        }
    }
    if (gradients != nullptr) {
        delete gradients;
    }
    if (values != nullptr) {
        delete values;
    }
}

/* Output the weights of the layer */
void NN::Layer::print_weights() {
    for (int i = 0; i < weights.size(); i++) {
        std::cout << "Node " << i + 1 << ": ";
        for (int j = 0; j < weights[i]->size(); j++) {
            std::cout << weights[i]->at(j) << " ";
        }
        std::cout << std::endl;
    }
}
/* Output the gradients of the layer */
void NN::Layer::print_gradients() {
    for (int i = 0; i < gradients->size(); i++)
            print(gradients->at(i));
}
/* Output the current values of the layer */
void NN::Layer::print_values() {
    for (int i = 0; i < values->size(); i++)
            print(values->at(i));
}