#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include "tensor.h"
#include "nn.h"
#include "functional.h"

namespace graph
{

    // data.pos: Node position matrix with shape [num_nodes, num_dimensions]

    typedef enum DataType
    {
        TRAIN,
        VAL,
        TEST
    } DataType;
    /**
     * @brief helper method to convert vectors of source and destination nodes to edge list representation
     * @param source(type vector<int>)
     * @param destination(type vector<int>)
     * @return edge list tensor (type tensor<int>) shape[2, num_edges]
     */
    cyg::tptr<int> vec_to_edge_list(std::vector<int> source, std::vector<int> destination);
    /**
     * edge list to adjancency matrix
     * @param edge_index(type tensor<int>) shape [2, *]
     * @param edge_attr(type tensor<float>) shape [*] Optional
     * @param n_nodes(tytpe int) Optional if not specified, the max value in edge_index will be used
     * @return adj_mat (type tensor<float>) shape [num_nodes, num_nodes]
     * fusing adj_mat with weight_mat so we have tensor<float> instead of int
     */
    cyg::tptr<float> edge_to_adj_mat(const cyg::tensor<int> &edge_index, cyg::tensor<float> *edge_attr = nullptr, size_t n_nodes = 0);

    /***
     * @brief adjacency matrix to edge list and edge attr if mat is filled with edge weight
     * @param adj_mat (type std::shared_ptr<float>) //
     * @return tuple(edge_index, edge_attr) shapes [2, num_edges] [num_edges]
     */
    std::tuple<cyg::tptr<int>, cyg::tptr<float>> adj_to_edge_list(cyg::tensor<float> &adj_mat);

    /**
     * @brief add self loops to graph, given the edge list rep
     * @param edge_index(type: tensor<int>)
     */
    std::tuple<cyg::tptr<int>, cyg::tptr<float>> add_self_loops(const cyg::tensor<int> &edge_index, cyg::tensor<float> *edge_attr = nullptr, const float &fillValue = 0, const int &num_nodes = 0);

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
        Data(const cyg::tptr<float> &x, cyg::tensor<int> *edge_index = nullptr, cyg::tptr<float> edge_attr = nullptr, cyg::tensor<float> *y = nullptr);
        /**
         * @brief returns a edge list rep, can be null if none has been set
         * this is usually preferred to adj matrix for rep graph data since it req less memory (and of course a bit more compact :=) )
         */
        cyg::tensor<int> *edge_index();

        void set_edge_index(cyg::tensor<int> *edge_index, cyg::tptr<float> edge_attr = nullptr);
        /**
         * @brief generates the fused adj matrix from the edge list rep of this data object
         * generate matrix on the fly
         * @return adj_mat (type std::shared_ptr<float>) shape [n_nodes, n_nodes]
         */
        cyg::tptr<float> to_adj();

        size_t num_nodes() const { return _num_nodes; }
        size_t num_node_features() const { return _num_node_features; };
        size_t num_edges() const { return _num_edges; };
        size_t num_edge_features() const { return _num_edge_features; };
        cyg::tptr<float> x() const { return _x; };
        cyg::tensor<int>* edge_index() const { return _edge_index; };
        cyg::tptr<float> edge_attr() const { return _edge_attr;};
        /**
         * @param mask(type std::shared_ptr<bool>)
         * @param size_t
         */
        void set_mask(cyg::tensor<bool> &mask, DataType type = DataType::TRAIN);

    cyg::tensor<bool> *_train_mask = nullptr;
    cyg::tensor<bool> *_val_mask = nullptr;
    cyg::tensor<bool> *_test_mask = nullptr;
    size_t _num_nodes, _num_node_features, _num_edges, _num_edge_features;
    cyg::tensor<int> *_adj_matrix = nullptr;
    cyg::tensor<int> *_edge_index = nullptr;
    cyg::tensor<float> *_y = nullptr;
    cyg::tptr<float> _x, _edge_attr;
    };

    class MessagePassing : public nn::Module
    {
    public:
        MessagePassing() {}
        virtual cyg::tptr<float> message(const cyg::tptr<float> &x, const cyg::tptr<float>* others=nullptr){ return  x; };
        virtual cyg::tptr<float> aggregate_and_update(const cyg::tptr<float> &x, const cyg::tensor<int> &edge_index) { throw std::runtime_error("not yet implemented"); };
        template<typename... T> cyg::tptr<float> operator()(T&...input) { return forward(std::forward<T>(input)...); };//std::forward
        virtual cyg::tptr<float> forward(Data &&input) { throw std::runtime_error("not yet implemented");}
        virtual cyg::tptr<float> forward(Data &&input, Data &input2) { throw std::runtime_error("not yet implemented");}
        cyg::tptr<float> propagate(const cyg::tensor<int> &edge_index, const cyg::tptr<float> &x, const cyg::tptr<float> &others);
    };

    // https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html#the-messagepassing-base-class
    class GCNConv : public MessagePassing
    {
        public:
            GCNConv(size_t in_channels, size_t out_channels, float dropout=0.0);
            /**
             * x shape [num_nodes, num_node_features]
             * edge_index shape [2, num_edges]
             */
            cyg::tptr<float> forward(Data && input) override;
            cyg::tptr<float> message(const cyg::tptr<float> &x, const cyg::tptr<float>* norm) override { return *norm * (*get_module("drop"))(x); } //  num_nodes * 1  *   num_nodes * num_node_features
            cyg::tptr<float> aggregate_and_update(const cyg::tptr<float> &x, const cyg::tensor<int> &edge_index) override;

        size_t _in_channels, _out_channels;
        float _dropout;
    };
}
#endif