#ifndef NN_H
#define NN_H
#include <iostream>
#include <vector>

/* Implementation of a simple neural network to play the Wumpus game */
/* Some code based off of: https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/ */

// use typedefs for future ease for changing data types like: float to double
typedef float Scalar;
typedef std::vector<Scalar> vec;

class NN {
public:
    /* Constructor: The topology vector describes how many neurons we have 
    in each layer, and the size of this vector is equal to a number of layers 
    in the neural network. 
    Topology refers to each layer, and number of neurons per layer */
    NN(std::vector<uint> topology, Scalar learning_rate);
    ~NN();
    /* Forward propogation of data through neural network to get weights */
    void forward(vec input);
    /* Backwards propogation of error */
    void backprop(Scalar output);
    /* Error calculation */
    void loss(Scalar output);
    /* Update weights */
    void updateWeights();
    /* To train given a dataaset */
    void train(std::vector <vec> input, vec output);
    /* Mathematical Functions */
    Scalar dot_product(vec a, vec b);
    Scalar activationFunction(Scalar x);
    Scalar activationFunctionDerivative(Scalar x);

    struct Layer {
        struct Node {
            vec weights;
            vec gradients;
            Scalar value;

            Node();
            Node(int next_num_nodes);
            ~Node();

            void clear_gradients();
        };
        struct BiasNode : public Node {
            BiasNode(int next_num_nodes);
            ~BiasNode();
        };
        std::vector <Node *> nodes;
        int num_nodes, next_num_nodes;

        Layer();
        Layer(int num_nodes, int next_num_nodes);
        ~Layer();

        void clear_gradients();

        void print_weights();
        void print_gradients();
        void print_values();
    };

    Scalar learning_rate;
    std::vector<uint> topology;
    std::vector<Layer *> layers;
};

#endif

