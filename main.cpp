#include "nn.h"

int main() {

    srand(time(NULL));

    NN n({6,5,1}, 0.005);

    // n.layers[0]->print_values();
    // n.layers[0]->print_weights();

    std::vector<vec> inputs = {{2,3,4,5,1,3},
                               {1,2,3,4,5,4},
                               {2,3,1,2,3,1},
                               {1,2,4,5,6,2}};
    vec outputs = {18, 19, 12, 20};

    n.train(inputs, outputs);

    // n.layers[0]->print_values();
    std::cout << "Finished" << std::endl;

    return 0;
}