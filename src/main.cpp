#include "../include/graph.h"
#include "tensor.h"

using namespace cyg;
using namespace graph;
using namespace nn;
using namespace std;

class Model: public MessagePassing
{
    public:
        //n_layers should be about the graph diameter / using skip connections to mitigate the effect of oversmooting
        Model(size_t num_atoms, size_t embed_size, size_t out_channels, size_t p, bool bias, size_t n_layers=1): MessagePassing(){ 
            register_module("embed", new Embedding(num_atoms, embed_size)); // N * embed_size
            register_module("pre", new MLP(embed_size, {embed_size*2, embed_size}, bias, p)); //preprocessing => N * embed_size
            for(auto i=0; i<n_layers; i++){
                register_module("enc"+std::to_string(i+1), new GCNConv(embed_size, out_channels)); //
                embed_size = out_channels;
            };
            register_module("post", new MLP(out_channels*2, {out_channels*4, out_channels*2, out_channels}, bias, p)); //postprocessing
            register_module("drop", new Dropout(p));
            register_module("relu", new ReLU());
        };
        tptr<float> forward(const Data &p) {
                return tptr<float>();
        };
};

int main(void)
{
    /**
     * i will be training a simple model for binding affinity prediction - i love this :=)
     * each protein is a graph, and it comprises a number of amino acids
     * each amino acid consists of 4 backbone atoms and a set of sidechain atoms (i will be ignoring the sidechains and backbone for this task) existing in some 3D space
     * so for a graph representing a protein
     * n_i for every nodes in the grap representing each amino acid
     * each n_i has an embedding h_v
     * 
     * so x = num_nodes * embedding_size
     * 
     * this approach is oversimplified
     * 
     * using synthetic data to test my workflow
     * use the SKEMPI/PDBBind dataset 
     * 
     * batch * 2 * num_aa * 1               batch * 2 * 2 * num_edges?
     * batch * 2 * num_aa * embed_size      batch * 2 * 2 * num_edges?
     *  
     */
    return 0;
};