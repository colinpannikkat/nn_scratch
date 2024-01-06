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
        else
            layers.push_back(new Layer(topology[i], topology[i + 1]));
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
    vec values;
    std::vector <vec> weights;

    /* Set first layer to input */
    for (int i = 0; i < layers[0]->num_nodes; i++) {
        layers[0]->nodes[i]->value = input[i];
    }

    for (int i = 1; i < layers.size(); i++) {
        // Get weight and value vector from previous node
        for (int j = 0; j < layers[i]->num_nodes; j++) {
            values.push_back(layers[i - 1]->nodes[j]->value);
            weights.push_back(layers[i - 1]->nodes[j]->weights);
        }
        // Dot product between weight and value vector to get new value
        for (int j = 0; j < layers[i]->num_nodes; j++) {
            layers[i]->nodes[j]->value = dot_product(values, weights[j]);
            print(dot_product(values, weights[j]));
        }
        values.clear();
        weights.clear();
    }

    return;
}

void NN::backprop(Scalar output) {
    
    loss(output);
    updateWeights();

    return;
}

void NN::loss(Scalar output) {

    // Get last layer, and calculate loss
    Layer *last_layer = layers[topology.size() - 1];
    last_layer->clear_gradients();

    for (int i = 0; i < last_layer->num_nodes; i++) {
        last_layer->nodes[i]->gradients.push_back(output - last_layer->nodes.back()->value);
    }

    for (int i = topology.size() - 2; i > 0; i--) {

        layers[i]->nodes[i]->gradients.push_back(output - last_layer->nodes.back()->value);
    }
    

    return;
}

void NN::updateWeights() {

    /* To update the weights we must multiply the weight by the LR, gradient, and partial derivative of loss */
    for (int i = 0; i < layers.size(); i++) {
        for (int j = 0; j < layers[i]->nodes.size(); j++) {
            Layer::Node *curr = layers[i]->nodes[j];
            for (int k = 0; k < curr->weights.size(); k++) {
                curr->weights[k] = curr->weights[k] * curr->gradients[k] * learning_rate;
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
        std::cout << "Actual output: ";
        layers[topology.size() - 1]->print_values();
        std::cout << "Expected output: " << output[i] << std::endl;
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


Scalar NN::activationFunction(Scalar x) { return std::tanhf(x); }
 
Scalar NN::activationFunctionDerivative(Scalar x) { return 1 - std::tanhf(x) * std::tanhf(x); }

NN::Layer::Node::Node() {

}

NN::Layer::Node::Node(int next_num_nodes) {
    // initialize with random weights
    for (int i = 0; i < next_num_nodes; i++) {
        weights.push_back((float)rand() / RAND_MAX);
    }
}

NN::Layer::Node::~Node() {
    
}

void NN::Layer::Node::clear_gradients() {
    if (gradients.size() > 0)
        gradients.clear();
}

NN::Layer::BiasNode::BiasNode(int next_num_nodes) {
    this->value = 1;
    for (int i = 0; i < next_num_nodes; i++)
        weights.push_back(1);
}

NN::Layer::BiasNode::~BiasNode() {
    
}
/* Default constructor for Layer */
NN::Layer::Layer() {

}

/* Constructor for Layer */
NN::Layer::Layer(int num_nodes, int next_num_nodes) {
    this->num_nodes = num_nodes;
    this->next_num_nodes = next_num_nodes;

    // create nodes
    for (int i = 0; i < num_nodes; i++) {
        nodes.push_back(new Node(next_num_nodes));
        // if not output node, add bias node
        if (next_num_nodes != 0) {
            nodes.push_back(new BiasNode(next_num_nodes));
        }
    }
}
/* Destructor for Layer */
NN::Layer::~Layer() {
    
}

void NN::Layer::clear_gradients() {
    for (int i = 0; i < num_nodes; i++) {
        nodes[i]->clear_gradients();
    }
}

/* Output the weights of the layer */
void NN::Layer::print_weights() {
    for (int i = 0; i < nodes.size(); i++) {
        std::cout << "Node " << i + 1 << ": ";
        for (int j = 0; j < nodes[i]->weights.size(); j++) {
            std::cout << nodes[i]->weights.at(j) << " ";
        }
        std::cout << std::endl;
    }
}
/* Output the gradients of the layer */
void NN::Layer::print_gradients() {
    for (int i = 0; i < nodes.size(); i++) {
        std::cout << "Node " << i + 1 << ": ";
        for (int j = 0; j < nodes[i]->gradients.size(); j++) {
            std::cout << nodes[i]->gradients.at(j) << " ";
        }
        std::cout << std::endl;
    }
}
/* Output the current values of the layer */
void NN::Layer::print_values() {
    for (int i = 0; i < nodes.size(); i++)
        print(nodes[i]->value);
}