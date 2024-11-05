#include "tensor.h"
#include <memory>
#include <array>
#include <valarray>
#include <iostream>

using namespace std;
using namespace cyg;

int main(void)
{
    cout<<"testing tensor initialization"<<endl;
    vector<int> dims = {2,3};
    auto arr1 = initialize<float>(dims, 3);
    // auto arr2 = initialize<float>(dims, 5);
    // auto zeros_arr = initialize<float>(dims, 0);
    auto t1 = make_shared<tensor<float>>(*arr1, dims, Device::cpu, false);

    // auto data = initialize<float>({3,4}, 2);
    // // dims.push_back(1);
    // // for(auto j:dims)cout<<j<<endl;
    // for(const auto& m:*data) cout<<m<<"\t";
    // vector<int> dims = {3,4,5};
    // auto t3 =  cyg::randn(dims);
    // // t3.unsqueeze(0);
    // // CHECK(t3.shape().sum()==dims3.sum());

    // // auto t3 = cyg::randn(dims, -1, 1);
    // for(auto m:*t3->data())cout<<m<<"\t";

    // auto t_ones = cyg::ones(dims, Device::cpu, false);
    // vector<int> d = {1,3};
    // auto t_zeros = cyg::ones(d, Device::cpu, true);
    // // auto dims = t_ones->shape();
    // // dims.push_back(1);

    // t_ones += t_zeros;
    // t_new= t_new + t_zeros;
    // t_new= t_new + t_zeros;
    // // auto t_new = t_ones->mean(t_zeros);
    // // cout<<"randn+ones \n"<<*t_new<<endl;
    // // cout<<"randn+ones \n"<<*t_zeros<<endl;
    // // t_new->grad_fn->backward(initialize(t_new->shape(), 1));
    // t_new->backward(initialize(dims, 1));
    // std::cout <<std::cout.precision(10);
    // cout<<*t_ones<<endl;
    // for(auto j: *t_ones->get_grad()) cout<<j<<"\t";
    // cout<<endl;
    // for(auto j: *t_zeros->get_grad()) cout<<j<<"\t";
    // cout<<endl;
    // cout<<"randn+ones \n"<<*t_new<<endl;

    // free(lhs.get());
    // free(output.get());
    // free(rhs.get());

    // free(add.get());
    // add.reset();
    // lhs.reset();
    // rhs.reset();
    // auto res = lhs + rhs;
    // auto res_vec = lhs + rhs;
    // res_vec->backward(ones(dims).get());

    // cout<<*res_vec<<endl;
    // cout<<*rhs<<endl
    // const size_t n_elements = 6;
    // const vector<int> dims = {2, 3};
    // const vector<float> zeros(n_elements, 0);
    // const vector<float> dat(n_elements, 1);
    // auto t_ones = cyg::ones(dims);
    // auto t_zeros = cyg::ones(dims);
    // auto t_rand = cyg::randn(dims);

    // // const vector<int> dims = {1,2};
    // // auto t = cyg::zeros(dims);
    // // t(0,0,0);//shoudl throw excep
    // // cout<<t<<endl;
    // t_rand.enable_grad(true);
    // cout<<t_rand.current_node<<endl;
    // t_rand.add(t_ones);
    // cout<<t_rand.current_node<<endl;
    // t_rand.backward(nullptr);
    // // for (auto j : *t_new.get_grad())
    // //     cout << j << "\t";
    // // for (auto j : *t_ones.get_grad())
    // //     cout << j << "\t";
    // for (auto j : *t_rand.get_grad())
    //     cout << j << "\t";
    // cout << endl;

    // const int size = 5;
    // vector<float> k(size);
    // iota(k.begin(), k.end(), 2);
    // // for(auto m:k)
    // //     cout<<m<<"\t";
    // cout<<k.size()<<endl;

    return 0;
};