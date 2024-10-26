#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"
#include <memory>
#include <vector>
#include <numeric>

using namespace std;
using namespace cyg;

const size_t n_elements = 6;
vector<int> dims = {2, 3};
vector<float> zeros_vec(n_elements, 0);
vector<float> ones_vec(n_elements, 1);
vector<float> ones_vec9(n_elements+3, 1);
auto t_ones = cyg::ones(dims, Device::cpu, true);
auto t_zeros = cyg::ones(dims, Device::cpu, true);
auto t_rand =cyg::randn(dims);
auto vec = initialize(dims, 3);
auto vec2 = initialize(dims, 5);
auto t_vec = make_shared<tensor>(*vec, dims, Device::cpu, true);
auto t_vec2 = make_shared<tensor>(*vec2, dims, Device::cpu, true);

// @todo write tests for all the added ops
TEST_CASE("testing cyg::ones")
{
    auto d = *t_vec->data();
    cout << "check tensor data" << "\n";
    CHECK(equal(d.cbegin(), d.cend(), vec->cbegin()));
    auto s = t_vec->n_elements();
    cout << "check tensor num of elements" << "\n";
    CHECK(s == n_elements);
    cout << "check tensor size" << "\n";
    CHECK(equal(dims.cbegin(), dims.cend(), t_vec->shape().cbegin()));
    cout << "check tensor rank" << "\n";
    CHECK(t_vec->rank() == dims.size());
    // cout<<*t_vec<<endl;
    cout<< "check tensor grad"<<"\n";
    t_ones->enable_grad(false);
    CHECK(t_ones->get_grad()==nullptr);
    t_ones->enable_grad(true);
    auto grad = *t_ones->get_grad();
    CHECK(equal(grad.cbegin(),grad.cend(), zeros_vec.begin()));
    cout << "check indexing operator" << "\n";
    CHECK((*t_ones)(0,0) != 2);
    CHECK((*t_ones)(0, 0) == 1);
    CHECK_THROWS((*t_ones)(0, 14));
    CHECK_THROWS((*t_ones)(12, 0));
    t_rand->enable_grad(true);
    // cout<<"randn \n" <<*t_rand<<endl;
    cout<<"testing backprop - Add"<<endl;
    auto t_new1 = t_ones / t_zeros;
    auto t_new = t_new1 / t_zeros;
    cout<<"randn+ones \n"<<*t_new<<endl;
    t_new->backward(initialize(dims, 1));
    for(auto j: *t_ones->get_grad()) cout<<j<<"\t";
    cout<<endl;
    for(auto j: *t_zeros->get_grad()) cout<<j<<"\t";
    cout<<endl;
    // cout<<*t_new<<endl;
    // for(auto j: *t_zeros->get_grad()) cout<<j<<"\t";
    // for(auto j: *t_rand->get_grad()) cout<<j<<"\t";
    // cout<<endl;
}