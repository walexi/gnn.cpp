#include "../include/graph.h"
#include "tensor.h"

using namespace cyg;
using namespace graph;
using namespace nn;
using namespace std;


class Model: public MessagePassing
{
    public:
        Model(size_t n_classes, size_t input_dim, std::vector<size_t> hidden_dims, size_t p, bool bias): MessagePassing(){ 
            register_module("pre", new MLP(input_dim, {input_dim*2, input_dim}, bias, p)); //preprocessing
            for(auto i=0; auto hidden_dim: hidden_dims){
                register_module("enc"+std::to_string(i+1), new GCNConv(input_dim, hidden_dim)); //
                input_dim = hidden_dim;
            };
            register_module("post", new MLP(input_dim, {input_dim*2, input_dim, n_classes}, bias, p)); //postprocessing
        };
        tptr<float> forward(const Data &x) {
                auto out = (*get_module("pre"))(x.x());
                for(auto i=0;i<_modules.size()-1;i++){
                    out = (*_modules[i+1].second)(out, x.edge_index());
                    out = nn::tanh(out);
                };
                out = (*get_module("post"))(out);
                return out;
        };
};

int main(void)
{
    
    return 0;
};
