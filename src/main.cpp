// #include "tdata.h"
#include "tensor.h"
#include <vector>
#include <iostream>
#include <numeric>
// using namespace tdata;
using namespace std;
using namespace cyg;

int main(void){
    const vector<int> dims = {1,2};
    auto t = cyg::zeros(dims);
    t(0,0,0);//shoudl throw excep
    cout<<t<<endl;
    // const int size = 5;
    // vector<float> k(size);
    // iota(k.begin(), k.end(), 2);
    // // for(auto m:k)
    // //     cout<<m<<"\t";
    // cout<<k.size()<<endl;
    

    return 0;
};