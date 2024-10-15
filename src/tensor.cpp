#include "tensor.h"
#include <iostream>
#include <random>
#include <numeric>
#include <ranges>
#include <algorithm>
#include <assert.h>

using namespace cyg;
using namespace std;

default_random_engine e(time(nullptr));
uniform_real_distribution<float> u(0, 1);

auto get_index(const tensor *t, std::vector<int> &dims)
{
    // dims = {1,2,3} n_elements=1*2*3=6 last_index in the flatten vec is {0,1,2} = 0*2 + 1*3=5
    transform(dims.cbegin(), dims.cend(), next(t->shape().cbegin()), dims.begin(), multiplies<int>()); // zip(dims[1:], this->dims[:-1]) zip(dims[1:]+[0], this->dims[:])
    auto index = accumulate(dims.cbegin(), dims.cend(), 0);                                            // sum transformation
    return index;
};

const float &tensor::operator()(int dim, ...)
{
    vector<int> dims = {dim};
    va_list args;
    va_start(args, dim);
    for (int i = 0; i < this->rank() - 1; ++i) // will chop if user enter more than tensor's rank
    {
        int idx = va_arg(args, int);
        dims.push_back(idx);
    }
    va_end(args);
    int i = 0;
    for (auto j : dims)
    {
        if (j >= this->shape()[i] || j < 0)
            throw runtime_error("invalid index");
        i++;
    }
    auto index = get_index(this, dims);
    cout << index << endl;
    return this->d[index];
};

std::ostream &cyg::operator<<(std::ostream &out, const tensor &d)
{
    out << "[ ( ";
    for (auto j : d.data())
        out << j << " ";
    out << "), size = ( ";
    for (auto r : d.shape())
        out << r << " ";
    out << "), requires_grad = ";
    string require_grad = (d.require_grad() == 0) ? "true" : "false";
    string device = (d.get_device() == Device::cpu) ? "cpu" : "cuda";
    out << require_grad << " device = " << device << " ]" << endl;
    return out;
}
tensor cyg::randn(const std::vector<int> &dims, Device device, bool requires_grad)
{
    auto v = initialize(dims, true);
    auto t = tensor(v, dims, device, requires_grad);
    return t;
}
tensor cyg::ones(const std::vector<int> &dims, Device device, bool requires_grad)
{
    auto v = initialize(dims, 1);
    auto t = tensor(v, dims, device, requires_grad);
    return t;
}
tensor cyg::zeros(const std::vector<int> &dims, Device device, bool requires_grad)
{
    auto v = initialize(dims, 0);
    auto t = tensor(v, dims, device, requires_grad);
    return t;
}
tensor cyg::ones_like(const tensor &a, Device device, bool requires_grad) { return ones(a.shape(), device, requires_grad); }
tensor cyg::zeros_like(const tensor &a, Device device, bool requires_grad) { return zeros(a.shape(), device, requires_grad); };
inline void cyg::move_tensor(tensor &tensor, Device device) {
    // tensor.set_device(Device::cpu);
    // move data from gpu to cpu
};
inline const vector<float> &cyg::initialize(const std::vector<int> &dims, int value, bool random)
{
    // assert(dims.size()==2); //enforce 2D for now
    auto d = accumulate(dims.cbegin(), dims.cend(), 1, multiplies<size_t>());
    vector<float> *v = new vector<float>(d, value); // make_unique<int>
    if (random)
        for (auto m : *v)
            m = u(e);
    // delete[] t;
    return *v;
};

void cyg::tensor::detach()
{
    move_tensor(*this, Device::cpu);
}

void cyg::tensor::enable_grad(bool requires_grad)
{
    if (requires_grad)
    {
        this->requires_grad = true;
        vector<float> *v = new vector<float>{0};
        this->grad = v;
    }
    else
    {
        this->requires_grad = false;
        this->grad = nullptr;
    };
}
void cyg::tensor::zero_grad() {
    // fill(this->grad.begin(), this->grad.end(), 0.0);
};

/**
 * tensor with require_grad=true are used in building a computational graph
 * for reverse mode
 * when we call backward, we have the current node of the tensor to be the last op
 * using the node, do a topological sort on the graph,
 * using the sorted nodes, simply run a loop, passing in the upstream grad down the line
 */
void cyg::tensor::backward(std::vector<float *> grad, std::vector<float *> z)
{
    // TODO: insert return statement here
    throw logic_error{"not yet implemented"};
};
cyg::tensor::tensor(std::vector<float> &ms, const std::vector<int> &dims, Device device, bool requires_grad) : d(ms), n_dims(dims.size()), dims(dims), device(device), requires_grad(requires_grad)
{
    auto d = accumulate(dims.cbegin(), dims.cend(), 1, multiplies<int>());
    if (ms.size() != d)
    { // check number of elements
        assert("out of range access");
        throw std::runtime_error{"out of bound range"};
    };
    if (requires_grad)
    {
        vector<float> *v = new vector<float>{0};
        this->grad = v;
        // this->grad = make_unique<vector<float*>>(v);
    };
    // assert valid dims, and check against the num of elements in ms
}; // size_t cannot be -ve, so no need assert sanity check
// cyg::tensor::~tensor() noexcept
// {
//     // TODO: insert return statement here
//     throw logic_error{"not yet implemented"};
// };
// move constructor
// cyg::tensor::tensor(tensor &&other)
// {
//     throw logic_error{"not yet implemented"};
// };
// //move assignment constructor
// tensor &cyg::tensor::operator=(tensor &&other)
// {
//     // TODO: insert return statement here
//     return *this;
// };
tensor &cyg::tensor::add(tensor &other)
{
    transform(other.data().begin(), other.data().end(), this->d.begin(), this->d.begin(), std::plus<float>());
    // auto ops = cyg::ops();
    // ops.func = [](vector<float> a,vector<float> b) {return a;};
    // this->current_node->parent

    return *this;
};

tensor &cyg::tensor::mul(const tensor &other)
{
    transform(other.data().begin(), other.data().end(), this->d.begin(), this->d.begin(), std::multiplies<float>());
    return *this;
};
tensor &cyg::tensor::sub(tensor &other)
{
    transform(this->d.begin(), this->d.end(), other.d.begin(), this->d.begin(), std::minus<float>());
    return *this;
};
tensor &cyg::tensor::div(tensor &other)
{
    transform(other.data().begin(), other.data().end(), this->d.begin(), this->d.begin(), std::divides<float>());
    return *this;
};
tensor &cyg::tensor::pow(const tensor &other)
{
    throw logic_error{"not yet implemented"};
};
tensor &cyg::tensor::lt(const tensor &other) const
{
    // TODO: insert return statement here
    throw logic_error{"not yet implemented"};
};