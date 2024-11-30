#ifndef GRAPH_H
#define GRAPH_H

#include "tensor.h"
#include "nn.h"
#include "functional.h"
namespace graph 
{

// data.pos: Node position matrix with shape [num_nodes, num_dimensions]

    template<class T>
    using tptr = std::shared_ptr<cyg::tensor<T>>;


    using namespace std; //not a good practice, but just for the sake of brevity :-)
    using namespace cyg;

    typedef enum DataType{
        TRAIN,
        VAL,
        TEST
    } DataType;
    /**
     * @brief helper method to convert vectors of source and destination nodes to edge list representation
     * @param source(type vector<int>)
     * @param destination(type vector<int>)
     * 
     * @return edge list tensor (type tensor<int>) shape[2, num_edges]
     */
    tptr<int> vec_to_edge_list(vector<int> source, vector<int> destination){
        if(source.size()!=destination.size()) throw runtime_error("input vectors must be of same length");
        auto num_edges = source.size();
        source.insert(source.end(), destination.begin(), destination.end());
        valarray<int>* data = new valarray<int>(source.data(), source.size());
        vector<size_t> dims = {2, num_edges};
        return make_shared<tensor<int>>(dims, data, false);
    }
    
    /**
     * edge list to adjancency matrix
     * @param edge_index(type tensor<int>) shape [2, *]
     * @param edge_attr(type tensor<float>) shape [*] Optional
     * @param n_nodes(tytpe int) Optional if not specified, the max value in edge_index will be used
     * 
     * @return adj_mat (type tensor<float>) shape [num_nodes, num_nodes]
     * fusing adj_mat with weight_mat so we have tensor<float> instead of int
     */
    tptr<float> edge_to_adj_mat(const cyg::tensor<int> &edge_index, tensor<float>* edge_attr = nullptr, size_t n_nodes=0){
        if(edge_attr!=nullptr && edge_index.shape()[1]!=edge_attr->shape()[0]) throw runtime_error("invalid inputs, number of edges in edge_index must be equal to size of edge_attr");
        size_t num_nodes = get<0>(functional::max<int>(edge_index))->item();
        if(n_nodes!=0) num_nodes = n_nodes;
        auto new_data = new valarray<float>(num_nodes*num_nodes);
        auto dims = {num_nodes, num_nodes};
        auto num_edges = edge_index.shape()[1];
        for(auto i=0; i<num_edges; i++){
            //  auto r = data[i];
            // auto c = data[i+num_edges];
            int r = edge_index(0, i);
            int c = edge_index(1, i);
            float w = 1;
            if(edge_attr!=nullptr) w = (*edge_attr)[i]; //indexiing 1D tensor
            (*new_data)[r*num_nodes + c]=w;
        }

        return make_shared<tensor<float>>(dims, new_data, false);

    };

    /***
     * @brief adjacency matrix to edge list and edge attr if mat is filled with edge weight
     * 
     * @param adj_mat (type std::shared_ptr<float>) //
     * 
     * @return tuple(edge_index, edge_attr) shapes [2, num_edges] [num_edges]
     * 
     */
    std::tuple<tptr<int>, tptr<float>> adj_to_edge_list(tensor<float> &adj_mat){
        auto data = (*adj_mat.data());
        auto n_nodes = adj_mat.shape()[0]; //check its a square matrix
        vector<int> src, dst; //src=rows dst=cols
        vector<float> w_mat;
        for(auto i=0; i<data.size(); i++){
            if(int(data[i])!=0){
                src.push_back(i/n_nodes);
                dst.push_back(i%n_nodes);
                w_mat.push_back(data[i]);
            }
        }
        src.insert(src.end(), dst.begin(), dst.end());
        auto edge_id_data = new valarray<int>(src.data(), src.size());
        auto edge_attr_data = new valarray<float>(w_mat.data(), w_mat.size());
        vector<size_t> dims = {dst.size()};
        auto edge_attr = make_shared<tensor<float>>(dims, edge_attr_data, false);
        dims.insert(dims.begin()+0, 2);
        auto edge_index = make_shared<tensor<int>>(dims, edge_id_data, false);

        return {edge_index, edge_attr};
    }

    /**
     * @brief add self loops to graph, given the edge list rep
     * 
     * @param edge_index(type: tensor<int>)
     */
    std::tuple<tptr<int>, tptr<float>> add_self_loops(const tensor<int> &edge_index, tensor<float>* edge_attr=nullptr, const float& fillValue=0, const int& num_nodes=0){
        
        auto mat = edge_to_adj_mat(edge_index, edge_attr, num_nodes);
        mat->fill_diagonal_(fillValue);

        return adj_to_edge_list(*mat);
    }
    
    /**
     * class to handle graph data
     */
    class Data
    {
        public:
            /**
             * @param x(type tensor<float>) shape [num_node, num_node_features]
             * @param edge_index(type tensor<int>) shape[2, num_edges] coo format
             * @param edge_attr(type tensor<float>) shape[num_edges, num_edge_features]
             * @param y(type tensor<float>)
             */
            Data(const tptr<float>& x, tensor<int>* edge_index=nullptr, tptr<float> edge_attr=nullptr, tensor<float>* y=nullptr): 
            _x(x), _edge_index(edge_index), _edge_attr(edge_attr), _y(y), _num_nodes(x->shape()[0]), _num_node_features(x->shape()[1])
            {
                // x, edge_index and edge_attr must be 2D
                if(x->rank()!=2) throw runtime_error("invalid input for x, must be 2D");
                if(edge_index!=nullptr){
                    if(edge_index->rank()!=2 || edge_index->shape()[0]!=2) throw runtime_error("invalid input for x, must be of 2D");
                    // num_nodes (x->shape()[0]) should be greater than the max value in edge_index
                    _num_edges = edge_index->shape()[1];
                    if(x->shape()[0] <= get<0>(edge_index->max())->item()) throw std::runtime_error("invalid input, max value in edge_index should be less than the number of nodes from x");
                    if(edge_attr!=nullptr){
                        if(edge_attr->rank()!=2) throw std::runtime_error("pls check input tensors, must of 2D for x, edge_index and edge_attr");
                        if(edge_index->shape()[1]!=edge_attr->shape()[0]) throw runtime_error("invalid edge_index and/or edge_attr input, edge_index should of [2, num_edges] and edge_attr should be of [num_edges, num_edge_feature]");
                        _num_edge_features = edge_index->shape()[0];
                    }
                }
            }
            /**
             * @brief returns a edge list rep, can be null if none has been set
             * this is usually preferred to adj matrix for rep graph data since it req less memory (and of course a bit more compact :=) )
             */
            tensor<int>* edge_index(){
                if(_edge_index==nullptr){
                    // TODO replace with logging warn
                    throw std::runtime_error("pls provide adj matr or edge");
                }
                return _edge_index;
            }

            void set_edge_index( tensor<int>* edge_index, tptr<float> edge_attr = nullptr){
                delete _edge_index;
                _edge_index = edge_index;
                _edge_attr = edge_attr;
            }
            /**
             * @brief generates the fused adj matrix from the edge list rep of this data object
             * generate matrix on the fly
             * 
             * @return adj_mat (type std::shared_ptr<float>) shape [n_nodes, n_nodes]
             */
            tptr<float> to_adj(){
                if(_edge_index==nullptr) {
                    throw std::runtime_error("pls provide adj matr or edge");
                }
                auto dims = {_num_nodes, _num_nodes};
                auto new_data = std::valarray<float>(0.0, _num_nodes*_num_nodes);
                auto adj_mat = edge_to_adj_mat(*_edge_index, _edge_attr.get(), _num_nodes);
                
                return adj_mat;
            }
            size_t num_nodes(){
                return _num_nodes;
            }
            size_t num_node_features(){
                return _num_node_features;
            }
            size_t num_edges(){
                return _num_edges;
            }
            size_t num_edge_features(){
                return _num_edge_features;
            }
            /**
             * @param mask(type std::shared_ptr<bool>)
             * @param size_t
             */
            void set_mask(tensor<bool>& mask, DataType type= DataType::TRAIN){
                if(mask.numel()!=_num_nodes) throw runtime_error("invalid input, mask must be 1D and of same size with num of nodes in graph");
                switch (type)
                {
                case 0:
                    delete _train_mask;
                    _train_mask = &mask;
                    break;
                case 1:
                    delete _val_mask;
                    _val_mask = &mask;
                    break;
                case 2:
                    delete _test_mask;
                    _test_mask = &mask;
                    break;
                default:
                    break;
                }
            }

        tensor<bool>* _train_mask = nullptr; tensor<bool>* _val_mask = nullptr; tensor<bool>* _test_mask = nullptr;
        size_t _num_nodes, _num_node_features, _num_edges, _num_edge_features;
        tensor<int>* _adj_matrix = nullptr; tensor<int>* _edge_index = nullptr; tensor<float>* _y = nullptr;
        tptr<float> _x, _edge_attr;
        
    };

 
    template<class T>
    class MessagePassing: public nn::Module
    {
        public:
            MessagePassing(const string aggr="add"){}
            virtual tptr<float> message(const tptr<float> &x, const tptr<float> &others){

                return x;
            };
            virtual tptr<float> update(const tptr<float> &x_t, const tptr<float> &x){
                // auto adj_mat = to_adj_mat(edge_index, nullptr, x->shape()[0]); // num_nodes * num_nodes
                // auto out = adj_mat + x; N*N + N*F
                return x_t + x;
            };
            virtual tptr<float> aggregate(const tensor<int> &edge_index, const tptr<float>& x){
                auto adj_mat = edge_to_adj_mat(edge_index, nullptr, x->shape()[0]); // num_nodes * num_nodes
                auto agg_x = adj_mat->mm(x);
                return agg_x;
            };
            /**
             * useful since i will be using the adj mat to aggregrate messages from neighbouring nodes
             * though more expensive, torch provides a sparsetensor impl which I may not be able to replicate here.
             * sparsetensor 
             */
            virtual tptr<float> aggregate_and_update(const tensor<int> &edge_index, const tptr<float> &x){
                return x;
            };

            tptr<float> propagate(const tensor<int> &edge_index, const tptr<float> &x, const tptr<float> &others){
                auto out = message(x, others);
                if(&T::aggregate_and_update == &MessagePassing::aggregate_and_update){ // https://stackoverflow.com/a/37588766/2317681
                    out = aggregate_and_update(edge_index, out);
                }else{
                    out = aggregate(edge_index, out);
                    out = update(out, x);
                }
                return out;
            }
    };

//https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
    class GCNConv: public MessagePassing<GCNConv>
    {
        public:
            GCNConv(size_t in_channels, size_t out_channels): MessagePassing<GCNConv>("add"), _in_channels(in_channels), _out_channels(out_channels){
                    register_module("lin", new nn::Linear(in_channels, out_channels, false));
                    auto dims = {out_channels};
                    register_parameter("bias", std::make_shared<tensor<float>>(dims, 0, true));
            }

            /**
             * x shape [num_nodes, num_node_features]
             * edge_index shape [2, num_edges]
             */
            tptr<float> forward(const tptr<float> &x, tensor<int>* edge_index_) override{
                auto [edge_index, _] = add_self_loops(*edge_index_, nullptr, 0, x->shape()[0]);
                auto out = (*get_module("lin"))(x);
                auto adj_mat = edge_to_adj_mat(*edge_index, nullptr, x->shape()[0]);
                adj_mat->sum(-1, true, true); // get the degree nodes => N*1
                auto norm  = adj_mat->pow(-0.5); // sqrt(deg(i))
                out = propagate(*edge_index, out, norm);
                out = out + get_parameter("bias");

                return out;
            }
            tptr<float> message(const tptr<float>& x, const tptr<float>& norm) override{
                return norm * x; //  num_nodes * 1  *   num_nodes * num_node_features
            }
            tptr<float> aggregate_and_update(const tensor<int> &edge_index, const tptr<float> &x) override{
                auto adj_mat = edge_to_adj_mat(edge_index, nullptr, x->shape()[0]); // num_nodes * num_nodes
                auto agg_x = adj_mat->mm(x); // num_nodes * num_nodes o num_nodes * num_nodes_feature => num_nodes * num_nodes_feature
                // agg_x = agg_x + x; //implm of self loop in forward precludes the need for this step
                return agg_x;
            };

        size_t _in_channels, _out_channels;
    };
//     // class GraphSAGE: public MessagePassing
//     // {
//     //     public:
//     //         GraphSAGE(int in_dim, int out_dim): MessagePassing(){
//     //             register_module("lin1", new nn::Linear(in_dim, out_dim, true));
//     //             register_module("lin2", new nn::Linear(in_dim, out_dim, true));   
//     //         }
//     //         void message_passing(const std::shared_ptr<cyg::tensor<double>> input_tensor, const std::shared_ptr<cyg::tensor<int>> edge_index);

//     // }
}
#endif