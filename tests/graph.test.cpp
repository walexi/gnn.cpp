#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "graph.h"
#include "nn.h"
#include <memory>
#include <vector>

using namespace std;
using namespace graph;
using namespace cyg;

const float tol = 1e-40;
const auto compare = [](float a){ return fabs(a)<=tol;};


TEST_CASE("testing g")
{
    // see graph under scribbles
    auto src = {1, 2, 3, 0, 4, 1, 2, 3};
    auto dst = {1, 2, 0, 1, 2, 2, 1, 1};
    auto edge_list = vec_to_edge_list(src, dst);
    std::cout<<*edge_list<<"\n";

    auto mat = edge_to_adj_mat(*edge_list);
    std::cout<<*mat->to_int()<<"\n";

    auto [el, ew] = adj_to_edge_list(*mat);
    std::cout<<*el<<"\n";
    std::cout<<*ew<<"\n";

    auto t = tensor({2,3}, 0, false);
    std::cout<<std::get<1>(t.max())->item()<<"\n";
    auto n_nodes = 15;
    auto nodes_feat = 10;
    auto x = randn({15, 10}, 0, 2, true);
    auto data = Data(x, edge_list.get());
    std::cout<<data.num_nodes()<<"\n";
    std::cout<<*data.to_adj()<<"\n";

    auto m = GCNConv(nodes_feat, 20);
    auto out = m(data);
    std::cout<<*out<<"\n";
};
