#define NDEBUG
#include <cassert>

#include "tensor.h"
#include "utils.h"
#include <memory>

using namespace cyg;
using namespace std;

// const long MAX_DIMENSION = 500;

// int flogRefCount = 1; // use to log tensors ref count, 0=yes 1= no logging

tptr<float> cyg::randn(vector<size_t> dims, int low, int high, bool requires_grad)
{
    assertm(low < high, "low must be lower than high, pls check your input params");
    if (low >= high)
        throw runtime_error("pls check input params, low must be lower than high");
    auto vec = initialize<float>(dims, 1);
    generate(begin(*vec), end(*vec), [&]()
             { return generate_random(low, high); });
    return make_shared<tensor<float>>(dims, vec, requires_grad);
}
void cyg::no_grad(vector<tptr<float>> ts)
{
    for (const auto &t : ts)
    {
        t->requires_grad_(false);
    }
}
void cyg::enable_grad(vector<tptr<float>> ts)
{
    for (const auto &t : ts)
    {
        t->requires_grad_(true);
    }
}
tptr<int> cyg::eye(size_t n, size_t m)
{
    if (m == INT_MAX)
        m = n;
    auto dims = {n, m};
    auto t = make_shared<tensor<int>>(dims, 0, false);
    t->fill_diagonal_(1);
    return t;
};