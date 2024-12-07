#include "../include/graph.h"
#include "tensor.h"

using namespace cyg;
using namespace graph;
using namespace nn;
using namespace std;

class Model: public MessagePassing
{
    public:
        //n_layers should be about the graph diameter, and i'll be using skip connections to mitigate the effect of oversmooting
        Model(size_t in_channels, size_t out_channels, size_t p, bool bias, size_t n_layers=1): MessagePassing(){ 
            // thinking of sharing params btw p1 & p2, definitely not a good idea
            register_module("pre", new MLP(in_channels, {in_channels*2, in_channels}, bias, p)); //preprocessing
            for(auto i=0; i<n_layers; i++){
                register_module("enc"+i+1, new GCNConv(in_channels, out_channels));
                in_channels = out_channels;
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
     * i will be representing proteins on an atomic level
     * each protein is a graph, and it comprises a number of amino acids
     * each amino acid consists of 4 backbone atoms and a set of sidechain atoms (i will be ignoring the sidechains and backbone for this task) existing in some 3D space
     * so for a graph representing a protein
     * i have n_i for every nodes in the grap representing each amino acid
     * each n_i has an embedding h_v
     * 
     * so x = num_nodes * embedding_size
     * where num_nodes is num of amino acids in the protein
     * 
     * the goal is to pass in the graphs of two proteins that form a complex into a model and output the binding affinity
     * 
     * we will be using the SKEMPI/PDBBind dataset which gives us mutations carried on protein to protein complexes and their corresponding energies (dissociation constant, etc)
     * 
     * we want our model to predict the binding affinty of the proteins we'll be designing -> lol ahaha
     * 
     * in a way i want to understand the dynamics of some pertubations to the structural state of a protein with respect to some property of interest (in this case binding affinity) -  a topic for some other good day
     * 
     * this approach is oversimplified, so dont do this at home, lol
     * 
     * using synthetic data to test my workflow
     * then  use the SKEMPI/PDBBind dataset 
     * 
     * i'm not taking the geometric features of the amino acids (contact residues in this case and the backbone/sidechain) into account
     * 
     */
    return 0;
};