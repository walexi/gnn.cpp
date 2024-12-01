#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define NDEBUG
#include <cassert>

#include "doctest.h"
#include "tensor.h"
#include "utils.h"
#include <memory>
#include <valarray>
#include <numeric>

using namespace std;
using namespace cyg;

const float tol = 1e-20;
const auto compare = [](float a){ return bool(fabs(a)<=tol);};


TEST_CASE("testing tensor")
{
    cout<<"testing tensor initialization"<<endl;
    vector<size_t> dims = {3, 6,9};
    auto arr1 = initialize<float>(dims, 3);
    auto arr2 = initialize<float>(dims, 5);
    auto zeros_arr = initialize<float>(dims, 0);
    auto arr_int = initialize<int>(dims, 3);
    
    CHECK_THROWS_WITH_AS(tensor<int>(dims, arr_int, true), ERROR_GRAD_DTYPE, std::runtime_error); // setting grad on non-float dypes tensors
    
    auto t1 = make_shared<tensor<float>>(dims, arr1, true);
    valarray<float> diff1 = *t1->data() - *arr2;
    CHECK_FALSE(all_of(begin(diff1), end(diff1), compare)); //check data
    
    valarray<float> diff2 = *t1->data() - *arr1;
    CHECK(all_of(begin(diff2), end(diff2), compare));
    
    vector<size_t> dims2 = {10,20};
    CHECK_THROWS_WITH_AS(tensor<float>(dims2, arr1, false), ERROR_SIZE_MISMATCH, std::runtime_error); //check sizemismatch

    cout<<"testing tensor grad"<<endl;
    t1->requires_grad_(true);
    valarray<float> diff3 = *t1->grad() - *zeros_arr;
    CHECK(all_of(begin(diff3), end(diff3), compare));
    t1->requires_grad_(false);
    CHECK(t1->grad()==nullptr);
    cout<<"testing squeeze and unsqueeze ops"<<endl;
    vector<size_t> dims3 = {1,1,1,2,3,4};
    auto t2 = make_shared<tensor<float>>(dims3, 1);
    t2->squeeze();
    vector<size_t> dims33 = {2,3,4};
    auto s = t2->shape();
    CHECK(t2->shape() == dims33);
    t2->unsqueeze(2);
    vector<size_t> dims4 = {2,3,1,4};    
    CHECK(t2->shape() == dims4);
    t2->squeeze();
    t2->unsqueeze(-2);
    CHECK(t2->shape() == dims4);
    CHECK_THROWS_AS(t2->unsqueeze(5), std::runtime_error);
    

    cout << "check indexing operator" << "\n";
    t2->squeeze(); //{2,3,4}
    CHECK_THROWS_AS((*t2)(1,2,10), std::runtime_error);
    CHECK((*t2)(0, 0, 0) == (*t2->data())[0]);

    cout<< "addding tensors"<<"\n";
    t1->requires_grad_(true);
    CHECK_THROWS_WITH_AS(t1 += 3, ERROR_IN_PLACE_OP_LEAF, std::runtime_error);


    auto lhs = cyg::randn({2,3}, 1, 3, true);
    auto rhs = cyg::randn({2, 3,6}, 1, 3, true);
    auto res = lhs->mm(rhs);
    lhs->unsqueeze(0);
    lhs->repeat(0, 5);
    // auto dd5 = dd4->sum();
    // res->backward(ones_like(res).get());
    cout<<*lhs<<endl;
    cout<<*res<<endl;
    cout<<*res->transpose(-1,-2)<<endl;
    res->requires_grad_(false);
    res->sum(-1, true, true);
    cout<<*res<<endl;
    // cout<<*lhs<<endl;
    // cout<<*rhs<<endl;
    // cout<<printND(*lhs->get_grad(), lhs->shape()).str()<<endl;
    // cout<<printND(*rhs->get_grad(), rhs->shape()).str()<<endl;
    // t1->unsqueeze(0);
    // // t1 * float(2);
    // CHECK_THROWS_WITH_AS(t1+=t2, ERROR_SIZE_MISMATCH, std::runtime_error);
    // auto t3 = cyg::randn(t2->shape(), -1, 1, true);
    // CHECK_THROWS_WITH_AS(t3+=t2, ERROR_IN_PLACE_OP_LEAF, std::runtime_error);
    
//     // cout<<"output here\n";
// //     cout<<*l<<endl;
// //     auto d = *t_vec->data();
// //     cout << "check tensor data" << "\n";
// //     CHECK(equal(d.cbegin(), d.cend(), vec->cbegin()));
// //     auto s = t_vec->n_elements();
// //     cout << "check tensor num of elements" << "\n";
// //     CHECK(s == n_elements);
// //     cout << "check tensor size" << "\n";
// //     CHECK(equal(dims.cbegin(), dims.cend(), t_vec->shape().cbegin()));
// //     cout << "check tensor rank" << "\n";
// //     CHECK(t_vec->rank() == dims.size());
// //     // cout<<*t_vec<<endl;
// //     
// //     cout<< "check mean"<<endl;
// //     cout<<*t_rand<<endl;
// //     cout<< "check mean"<<endl;
// //     // sth is wrong here @todo fix 
// //     cout<<*t_rand->mean(1, true)<<endl;
//     // cout<<"randn \n" <<*t_rand<<endl;
//     // cout<<"testing backprop - Add"<<endl;
//     // t_rand->enable_grad(true);
//     // auto t_new1 = t_ones / t_zeros;
    // auto t_new = t_new1 / t_zeros;
    // cout<<"randn+ones \n"<<*t_new<<endl;
    // t_new->backward(initialize(dims, 1));
    // for(auto j: *t_ones->get_grad()) cout<<j<<"\t";
    // cout<<endl;
    // for(auto j: *t_zeros->get_grad()) cout<<j<<"\t";
    // cout<<endl;
    // // cout<<*t_new<<endl;
    // for(auto j: *t_zeros->get_grad()) cout<<j<<"\t";
    // for(auto j: *t_rand->get_grad()) cout<<j<<"\t";
    // cout<<endl;
}