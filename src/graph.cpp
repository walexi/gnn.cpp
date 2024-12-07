#include <iostream>
#include "graph.h"

using namespace graph;
using namespace cyg;
using namespace nn;
using namespace functional;
using namespace std;

tptr<int> graph::vec_to_edge_list(vector<int> source, vector<int> destination)
{
    if (source.size() != destination.size())
        throw runtime_error("input vectors must be of same length");
    auto num_edges = source.size();
    source.insert(source.end(), destination.begin(), destination.end());
    valarray<int> *data = new valarray<int>(source.data(), source.size());
    vector<size_t> dims = {2, num_edges};
    return make_shared<tensor<int>>(dims, data, false);
};

tptr<float> graph::edge_to_adj_mat(const tensor<int> &edge_index, tensor<float> *edge_attr, size_t n_nodes)
{
    if (edge_attr != nullptr && edge_index.shape()[1] != edge_attr->shape()[0])
        throw runtime_error("invalid inputs, number of edges in edge_index must be equal to size of edge_attr");
    size_t num_nodes = get<0>(functional::max<int>(edge_index))->item();
    if (n_nodes != 0)
        num_nodes = n_nodes;
    auto new_data = new valarray<float>(num_nodes * num_nodes);
    auto dims = {num_nodes, num_nodes};
    auto num_edges = edge_index.shape()[1];
    for (auto i = 0; i < num_edges; i++)
    {
        //  auto r = data[i];
        // auto c = data[i+num_edges];
        int r = edge_index(0, i)->item();
        int c = edge_index(1, i)->item();
        float w = 1;
        if (edge_attr != nullptr)
            w = (*edge_attr)[i]; // indexiing 1D tensor
        (*new_data)[r * num_nodes + c] = w;
    }

    return make_shared<tensor<float>>(dims, new_data, false);
};

tuple<tptr<int>, tptr<float>> graph::adj_to_edge_list(tensor<float> &adj_mat)
{
    auto data = (*adj_mat.data());
    auto n_nodes = adj_mat.shape()[0]; // check its a square matrix
    vector<int> src, dst;              // src=rows dst=cols
    vector<float> w_mat;
    for (auto i = 0; i < data.size(); i++)
    {
        if (int(data[i]) != 0)
        {
            src.push_back(i / n_nodes);
            dst.push_back(i % n_nodes);
            w_mat.push_back(data[i]);
        }
    }
    auto edge_index = vec_to_edge_list(src, dst);
    auto edge_attr_data = new valarray<float>(w_mat.data(), w_mat.size());
    vector<size_t> dims = {dst.size()};
    auto edge_attr = make_shared<tensor<float>>(dims, edge_attr_data, false);

    return {edge_index, edge_attr};
}
tuple<tptr<int>, tptr<float>> graph::add_self_loops(const tensor<int> &edge_index, tensor<float> *edge_attr, const float &fillValue, const int &num_nodes)
{

    auto mat = edge_to_adj_mat(edge_index, edge_attr, num_nodes);
    mat->fill_diagonal_(fillValue);

    return adj_to_edge_list(*mat);
};

graph::Data::Data(const tptr<float> &x, tensor<int> *edge_index, tptr<float> edge_attr, tensor<float> *y)
    : _x(x), _edge_index(edge_index), _edge_attr(edge_attr), _y(y), _num_nodes(x->shape()[0]), _num_node_features(x->shape()[1])
{
    // x, edge_index and edge_attr must be 2D
    if (x->rank() != 2)
        throw runtime_error("invalid input for x, must be 2D");
    if (edge_index != nullptr)
    {
        if (edge_index->rank() != 2 || edge_index->shape()[0] != 2)
            throw runtime_error("invalid input for x, must be of 2D");
        // num_nodes (x->shape()[0]) should be greater than the max value in edge_index
        _num_edges = edge_index->shape()[1];
        if (x->shape()[0] <= get<0>(edge_index->max())->item())
            throw runtime_error("invalid input, max value in edge_index should be less than the number of nodes from x");
        if (edge_attr != nullptr)
        {
            if (edge_attr->rank() != 2)
                throw runtime_error("pls check input tensors, must of 2D for x, edge_index and edge_attr");
            if (edge_index->shape()[1] != edge_attr->shape()[0])
                throw runtime_error("invalid edge_index and/or edge_attr input, edge_index should of [2, num_edges] and edge_attr should be of [num_edges, num_edge_feature]");
            _num_edge_features = edge_index->shape()[0];
        }
    }
};

tensor<int> *graph::Data::edge_index()
{
    if (_edge_index == nullptr)
    {
        // TODO replace with logging warn
        throw runtime_error("pls provide adj matr or edge");
    }
    return _edge_index;
};

void graph::Data::set_edge_index(tensor<int> *edge_index, tptr<float> edge_attr)
{
    delete _edge_index;
    _edge_index = edge_index;
    _edge_attr = edge_attr;
}
tptr<float> graph::Data::to_adj()
{
    if (_edge_index == nullptr)
    {
        throw runtime_error("pls provide adj matr or edge");
    }
    auto dims = {_num_nodes, _num_nodes};
    auto new_data = valarray<float>(0.0, _num_nodes * _num_nodes);
    auto adj_mat = edge_to_adj_mat(*_edge_index, _edge_attr.get(), _num_nodes);

    return adj_mat;
};
void graph::Data::set_mask(tensor<bool> &mask, DataType type)
{
    if (mask.numel() != _num_nodes)
        throw runtime_error("invalid input, mask must be 1D and of same size with num of nodes in graph");
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
};

tptr<float> graph::MessagePassing::propagate(const tensor<int> &edge_index, const tptr<float> &x, const tptr<float> &others)
{
    auto out = message(x, &others);
    out = aggregate_and_update(out, edge_index);
    return out;
};

graph::GCNConv::GCNConv(size_t in_channels, size_t out_channels, float dropout) : MessagePassing(), _in_channels(in_channels), _out_channels(out_channels), _dropout(dropout)
{
    register_module("lin", new Linear(in_channels, out_channels, false));
    register_module("bnorm", new BatchNorm(out_channels));
    register_module("drop", new Dropout(dropout));
    register_module("relu", new ReLU());
    auto dims = {out_channels};
    register_parameter("bias", make_shared<tensor<float>>(dims, 0, true));
};

tptr<float> graph::GCNConv::forward(const Data &input)
{
    auto [edge_index, _] = add_self_loops(*input.edge_index(), nullptr, 0, input.num_nodes());
    auto out = (*get_module("lin"))(input._x); // num_nodes * out_channels
    out = (*get_module("bnorm"))(input._x);
    out = (*get_module("relu"))(input._x);
    auto adj_mat = edge_to_adj_mat(*edge_index, nullptr, input.num_nodes());
    auto deg = adj_mat->sum(-1, true) + 1; // get the nodes degree => N*1 add 1 for self loop
    /**
     * sqrt(deg(i)) * sqrt(deg(j))  for each j in the neighborhood of i
     * i.e i*j + i*k + i*l = i(j + k + l)
     * deg * adj_mat(wo self loop)
     * N*1 * N*N * N*1 = N * 1
     */
    deg = deg->pow(-0.5); // sqrt(deg(i)) check for inf N*1
    auto norm = adj_mat->mm(deg); // N*N o N*1 = N*1
    norm*=deg; // N*1 * N*1 = N*1
    out = propagate(*edge_index, out, norm); // num_nodes * out_channels
    out = out + get_parameter("bias");       // num_nodes * out_channels

    return out;
};

tptr<float> graph::GCNConv::aggregate_and_update(const tptr<float> &x, const tensor<int> &edge_index)
{
    // add agg is implemented here
    auto adj_mat = edge_to_adj_mat(edge_index, nullptr, x->shape()[0]); // num_nodes * num_nodes
    auto agg_x = adj_mat->mm(x);                                        // num_nodes * num_nodes o num_nodes * num_nodes_feature => num_nodes * num_nodes_feature
    // agg_x = agg_x + x; //implm of self loop in forward precludes the need for this step
    return agg_x;
};