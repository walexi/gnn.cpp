#include "../include/graph.h"
#include "tensor.h"

using namespace cyg;
using namespace graph;
using namespace nn;
using namespace std;

class Model: public MessagePassing
{
    public:
        //n_layers should be about the graph diameter / using skip connections to mitigate the effect of oversmoothing
        Model(size_t num_atoms, size_t embed_size, size_t out_channels, size_t p, bool bias, size_t n_layers=1): MessagePassing(){ 
            register_module("embed", new Embedding(num_atoms, embed_size)); // 2 * num_aa * embed_size for protein A and B
            register_module("pre", new MLP(embed_size, {embed_size*2, embed_size}, bias, p)); //preprocessing => 2 * num_aa * embed_size for protein A and B
            for(auto i=0; i<n_layers; i++){
                register_module("enc"+std::to_string(i+1), new GCNConv(embed_size, out_channels)); //
                embed_size = out_channels;
            }; // => 2 * num_aa * embed_size for proteins A and B
            //concat along feat => 2*num_aa * embed_size => 1
            register_module("post", new MLP(out_channels, {out_channels*2, out_channels, 1}, bias, p)); //postprocessing
        };
        //batch of 2 proteins
        tptr<float> forward(const Data &x) { //batch * 2 * num_aa => batch * 2 * num_aa * embed_size => batch * 2 o num_aa * embed_size
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