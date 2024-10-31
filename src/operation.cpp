#include "operation.h"
#include "tensor.h"
#include <memory>
#include <iostream>
#include <numeric>
#include <random>

using namespace std;
using namespace cyg;

default_random_engine re(time(nullptr));

/**
 * @brief resize to t1
 */
void resize(const int t1_size, valarray<float> *t2, const int& value=1)
{
    if (t1_size > t2->size())
        t2->resize(t1_size - t2->size(), value);
    // else
}

/**
 * forward and backward passes for basic operations such as Add, etc
 * inherits class Operation
 * it makes use of the Context<tensor> class for caching input tensors for backward computation
 * see the Context<tensor> class for more information.
 */
template <>
std::shared_ptr<tensor<float>> cyg::Add<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &lhs, const std::shared_ptr<tensor<float>> &rhs)
{
    assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
    if (lhs->get_device() != rhs->get_device())
        throw runtime_error("tensors are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    auto out_data = new (nothrow) std::valarray<float>(); // allocating object on the heap,dont forget
    if (out_data == nullptr)
        throw std::runtime_error("insufficient memory");
    *out_data = *lhs->data() + *rhs->data();
    auto out = make_shared<tensor<float>>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    if (lhs->require_grad() || rhs->require_grad())
        this->context->save_for_backward({lhs, rhs});

    return out;
}
template <>
void cyg::Add<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    string err_msg = "can backprop without a executing a forward computation first";
    assertm(var.size() != 0, err_msg);
    if (var.size() == 0)
        throw runtime_error(err_msg);
    for (const auto &t : var)
        if (t->require_grad())
        {
            resize(t->n_elements(), incoming_grad);
            t->backward(incoming_grad);
        }
    this->context.reset();
}
template <>
std::shared_ptr<tensor<float>> cyg::Mul<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &lhs, const std::shared_ptr<tensor<float>> &rhs)
{
    assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
    if (lhs->get_device() != rhs->get_device())
        throw runtime_error("tensors are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    auto out_data = new (nothrow) valarray<float>();
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    *out_data = *lhs->data() * *rhs->data();
    auto out = make_shared<tensor<float>>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    this->context->save_for_backward({lhs, rhs});
    return out;
}

template <>
void cyg::Mul<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 2, "invalid");
    auto lhs = var[0];
    auto rhs = var[1];
    resize(lhs->n_elements(), incoming_grad); // lhs same size with rhs
    if (rhs->require_grad())
    {
        valarray<float> grad = *incoming_grad * *lhs->data(); // y=a*b, dy/da = b = 1 * b
        rhs->backward(&grad);
    }
    if (lhs->require_grad())
    {
        valarray<float> grad = *incoming_grad * *rhs->data();
        lhs->backward(&grad);
    }
    this->context.reset();
}

template <>
std::shared_ptr<tensor<float>> cyg::Div<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &numerator, const std::shared_ptr<tensor<float>> &denominator)
{
    assert(numerator->get_device() == denominator->get_device() && "tensors are on different devices");
    if (numerator->get_device() != denominator->get_device())
        throw runtime_error("tensors are on different devices");
    auto req_grad = numerator->require_grad() || denominator->require_grad();
    auto out_data = new (nothrow) valarray<float>();
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    *out_data = *numerator->data() / *denominator->data();
    auto out = make_shared<tensor<float>>(*out_data, numerator->shape(), numerator->get_device(), req_grad);
    this->context->save_for_backward({numerator, denominator});

    return out;
}

template <>
void cyg::Div<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 2, "invalid"); // for debugging purpose
    auto numerator = var[0];
    auto denominator = var[1];
    resize(numerator->n_elements(), incoming_grad);
    // y= a/b y = a * b**-1   dy/da = b**-1=1/b  dy/db = -a*b**-2
    if (numerator->require_grad())
    {
        valarray<float> local_grad = *incoming_grad / *denominator->data();
        numerator->backward(&local_grad);
    }
    if (denominator->require_grad())
    { // dy/db = -a*b**-2
        valarray<float> local_grad = *incoming_grad * -1 * (*numerator->data() / std::pow(*denominator->data(), 2));
        denominator->backward(&local_grad);
    }
    this->context.reset();
}

template <>
std::shared_ptr<tensor<float>> cyg::Pow<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &base, const std::shared_ptr<tensor<float>> &exponent)
{
    assert(base->get_device() == exponent->get_device() && "tensors are on different devices");
    if (base->get_device() != exponent->get_device())
        throw runtime_error("tensors are on different devices");
    auto req_grad = base->require_grad() || exponent->require_grad();
    auto out_data = new (nothrow) valarray<float>();
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    *out_data = std::pow(*base->data(), *exponent->data());
    auto out = make_shared<tensor<float>>(*out_data, base->shape(), base->get_device(), req_grad);
    this->context->save_for_backward({base, exponent});

    return out;
}
template <>
void cyg::Pow<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 2, "invalid"); // for debugging purpose
    auto base = var[0];
    auto exponent = var[1];
    resize(base->n_elements(), incoming_grad);
    // y = b**e    dy/db = e*b**e-1
    if (base->require_grad())
    {
        valarray<float> local_grad = *incoming_grad * (*exponent->data() * std::pow(*base->data(), *exponent->data() - 1));
        base->backward(&local_grad);
    }
    if (exponent->require_grad())
    { // logy = elogb dy(1/y) = delogb dy/de = ylogb = b**e * logb     (logb natural log)
        valarray<float> local_grad = *incoming_grad * (std::pow(*base->data(), *exponent->data()) * std::log(*base->data()));
        exponent->backward(&local_grad);
    }
    this->context.reset();
}

inline tuple<valarray<size_t>, valarray<size_t>, valarray<size_t>> generate_idxs(const vector<size_t> tdims, const int &n_elements, const int &dim)
{
    valarray<size_t> strides(tdims.size());
    size_t s = 1;
    for (int i = tdims.size() - 1; i >= 0; --i)
    {
        strides[i] = s;
        s *= tdims[i];
    }
    valarray<size_t> sizes(tdims.data(), tdims.size());
    sizes[dim] = 1;
    valarray<size_t> id_data(n_elements);
    std::iota(begin(id_data), end(id_data), 0);
    const valarray<size_t> idxs = id_data[std::gslice(0, sizes, strides)];

    return {strides, sizes, idxs};
}
template <>
std::shared_ptr<tensor<float>> cyg::Mean<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &base, const int &dim, const bool &keepdim)
{
    CHECK_VALID_RANGE(dim, base->rank());
    valarray<size_t> strides, sizes, idxs;
    std::tie(strides, sizes, idxs) = generate_idxs(base->shape(), base->n_elements(), dim);
    valarray<float> data = *base->data();
    auto out_data = new (nothrow) valarray<float>(idxs.size());
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    //@todo improve using gslice
    for (int i = 0; i < idxs.size(); i++)
    {
        auto m = valarray(data[std::slice(idxs[i], base->shape()[dim], strides[dim])]).sum() / base->shape()[dim];
        (*out_data)[i] = m;
    };
    vector<size_t> new_dims;
    new_dims.assign(begin(sizes), end(sizes));

    auto output = make_shared<tensor<float>>(*out_data, new_dims, base->get_device(), base->require_grad());
    if (!keepdim)
        output->squeeze();

    this->context->save_for_backward({base});
    this->context->saved_data["dim"] = dim;

    return output;
}

template <>
void cyg::Mean<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 1, "invalid"); // for debugging purpose
    assertm(this->context->saved_data.size() == 2, "invalid");
    auto dim = this->context->saved_data["dim"];
    auto base = var[0];
    // y = (a_1 + a_2 + ... + a_n) / n    where n = base->shape()[dims]
    // dy/da_1 = 1/n
    auto out_data = new (nothrow) valarray<float>(base->n_elements(), 1 / base->shape()[dim]);
    resize(base->n_elements(), incoming_grad);

    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    if (base->require_grad())
    {
        valarray<float> local_grad = *incoming_grad * *out_data;
        base->backward(&local_grad);
    }
    this->context.reset();
}

template <>
std::shared_ptr<tensor<float>> cyg::Exp<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &t)
{
    auto out_data = new (nothrow) valarray<float>();
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");

    *out_data = std::exp(*t->data());

    auto out = make_shared<tensor<float>>(*out_data, t->shape(), t->get_device(), t->require_grad());
    this->context->save_for_backward({t});

    return out;
}

template <>
void cyg::Exp<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 1, "invalid"); // for debugging purpose
    // y= e**x   logy = xloge   logy = x  i/y(dy) = dx  dy/dx = y = e**x
    auto base = var[0];
    resize(base->n_elements(), incoming_grad);
    if (base->require_grad())
    {
        valarray<float> local_grad = *incoming_grad * std::exp(*base->data());
        base->backward(&local_grad);
    }
    this->context.reset();
}

template <>
std::shared_ptr<tensor<float>> cyg::Log<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &t)
{
    auto out_data = new (nothrow) valarray<float>();
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");

    *out_data = std::log(*t->data());

    auto out = make_shared<tensor<float>>(*out_data, t->shape(), t->get_device(), t->require_grad());
    this->context->save_for_backward({t});
    // this->context->saved_data["base"] = base;

    return out;
}

template <>
void cyg::Log<tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 1, "invalid");
    auto base = var[0];
    resize(base->n_elements(), incoming_grad);
    if (base->require_grad())
    { // y = logx   dy/dx = 1/x
        valarray<float> local_grad = *incoming_grad / *base->data();
        base->backward(&local_grad);
    }
    this->context.reset();
}
template <>
std::shared_ptr<tensor<float>> cyg::Sum<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &base, const int &dim, const bool &keepdim)
{
    CHECK_VALID_RANGE(dim, base->rank(), -1);
    auto out_data = new (nothrow) valarray<float>(1);
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    vector<size_t> new_dims = {1};
    if (dim == -1)
        (*out_data)[0] = (*base->data()).sum();
    else
    {
        valarray<size_t> strides, sizes, idxs;
        std::tie(strides, sizes, idxs) = generate_idxs(base->shape(), base->n_elements(), dim);
        valarray<float> data = *base->data();
        out_data->resize(idxs.size());
        if (out_data == nullptr)
            throw runtime_error("insufficient memory");
        //@todo improve using gslice
        for (int i = 0; i < idxs.size(); i++)
        {
            auto m = valarray(data[std::slice(idxs[i], base->shape()[dim], strides[dim])]).sum();
            (*out_data)[i] = m;
        };
        new_dims.assign(begin(sizes), end(sizes));
    }
    auto output = make_shared<tensor<float>>(*out_data, new_dims, base->get_device(), base->require_grad());
    if (!keepdim)
        output->squeeze();

    this->context->save_for_backward({base});
    this->context->saved_data["dim"] = dim;
    return output;
}

template <>
void cyg::Sum<cyg::tensor<float>>::backward(std::valarray<float> *incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size() == 1, "invalid");
    auto base = var[0];
    // y = a+b+c+d+.. dy/da = 1
    auto out_data = new (nothrow) valarray<float>(base->n_elements(), 1);
    if (out_data == nullptr)
        throw runtime_error("insufficient memory");
    resize(base->n_elements(), incoming_grad);
    if (base->require_grad())
    {
        valarray<float> local_grad = *incoming_grad * *out_data;
        base->backward(&local_grad);
    }
    this->context.reset();
}

// extern template class cyg::Sum<cyg::tensor<float>, float>;
// extern template std::shared_ptr<cyg::tensor<float>> cyg::Sum<cyg::tensor<float>, float>::forward(const std::shared_ptr<cyg::tensor<float>>&base, const int& dims, const bool& keepdim);
// extern template class cyg::Sum<cyg::tensor<int>, int>;
// extern template std::shared_ptr<cyg::tensor<int>> cyg::Sum<cyg::tensor<int>, int>::forward(const std::shared_ptr<cyg::tensor<int>>&base, const int& dims, const bool& keepdim);
// template class cyg::Add<cyg::tensor<float>>;
// template class cyg::Add<cyg::tensor<int>>;
// template std::shared_ptr<tensor<float>> cyg::Add<cyg::tensor<float>>::forward(const std::shared_ptr<tensor<float>> &lhs, const std::shared_ptr<tensor<float>> &rhs);
// template std::shared_ptr<tensor<int>> cyg::Add<cyg::tensor<int>>::forward(const std::shared_ptr<tensor<int>> &lhs, const std::shared_ptr<tensor<int>> &rhs);

// template void cyg::Add<cyg::tensor<float>>::backward(std::valarray<float>* incoming_gradeient);
// template void cyg::Add<cyg::tensor<int>>::backward(std::valarray<float>* incoming_gradeient);
