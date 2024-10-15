#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"
#include <vector>
#include <numeric>
// using namespace tdata;
using namespace std;
using namespace cyg;

const size_t n_elements = 2;
const vector<int> dims = {1,2};
const vector<float> dat(n_elements,1);
auto t_ones = cyg::ones(dims);
auto t_zeros = cyg::ones(dims);
auto t_rand = cyg::randn(dims);
vector<float> vec(n_elements, 3);
vector<float> vec2(n_elements, 5);
auto t_vec = tensor(vec, dims);
auto t_vec2 = tensor(vec2, dims);


TEST_CASE("testing cyg::ones") {
    auto d = t_ones.data();
    cout<<"check tensor data"<<"\n";
    CHECK(equal(d.cbegin(), d.cend(),dat.begin()));
    auto s = t_ones.n_elements();
    cout<<"check tensor num of elements"<<"\n";
    CHECK(s==n_elements);
    cout<<"check tensor size"<<"\n";
    CHECK(equal(dims.cbegin(), dims.cend(), t_ones.shape().begin()));
    cout<<"check tensor rank"<<"\n";
    CHECK(t_ones.rank()==dims.size());
    cout<<"check indexing operator"<<"\n";
    CHECK(t_ones(0,0)!=2);
    CHECK(t_ones(0,0)==1);
    CHECK_THROWS(t_ones(0,14));
    CHECK_THROWS(t_ones(12,0));
}
TEST_CASE("testing binary ops - add"){
    auto add_ = t_vec + t_vec2;
    vector<float> vecadd(n_elements, 8);
    cout<<"check adding two tensors"<<"\n";
    cout<<add_<<endl;
    CHECK(equal(add_.data().cbegin(), add_.data().cend(), vecadd.cbegin()));
    cout<<"check post ops effect on operands a"<<"\n";
    cout<<t_vec<<endl;
    CHECK(equal(t_vec.data().cbegin(), t_vec.data().cend(), vec.cbegin()));
    cout<<"check post ops effect on operands b"<<"\n";
    cout<<t_vec2<<endl;
    CHECK(equal(t_vec2.data().cbegin(), t_vec2.data().cend(), vec2.cbegin()));
}
TEST_CASE("testing binary ops - sub"){
    auto sub_ = t_vec2 - t_vec;
    vector<float> vecsub(n_elements, 2); //5-3
    cout<<"check sub two tensors"<<"\n";
    cout<<sub_<<endl;
    CHECK(equal(sub_.data().cbegin(), sub_.data().cend(), vecsub.cbegin()));
    cout<<"check post ops effect on operands a"<<"\n";
    CHECK(equal(t_vec.data().cbegin(), t_vec.data().cend(), vec.cbegin()));
    cout<<t_vec<<endl;
    cout<<"check post ops effect on operands b"<<"\n";
    CHECK(equal(t_vec2.data().cbegin(), t_vec2.data().cend(), vec2.cbegin()));
    cout<<t_vec2<<endl;

}