#include "tensor.h"
#include "operation.h"
#include <memory>
#include <iostream>
using namespace std;
using namespace cyg;
/**
 * 
 * std::valarray would have been helpful since its is very suited for vectorized operations on arrays of numbers
 * but i chose to stick with vectors
 * i tried valarray but realized it decays to pointers like array when passed as arguments to a function
 * i've been trying to avoid, the more reason why i didnt use arrays since i have to use template (template specialization)
 * which eventually increase the amount of code i have to write
 * 
 * may still consider using valarray but for now i'm sticking with vector for simplicity
 */
/**
 * lambda functions to handle basic operations for two scalars
 */
const auto op_add = [](float a, float b)
{ return a + b; };
const auto op_sub = [](float a, float b)
{ return a - b; };
const auto op_div = [](float a, float b)
{ return a / b; };
const auto op_mul = [](float a, float b)
{ return a * b; };
const auto op_pow = [](float a, float b)
{ return std::pow(a, b); };

/**
 * operator overloading for vector, can easily delegate the underlying operations to cuda if cuda is available
 * for speed up
 */
std::vector<float> cyg::pow(const std::vector<float> &v1, const float v2)
{
    auto v2_vec = new vector<float>(v1.size(), v2);
    return cyg::pow(v1, *v2_vec);
}

std::vector<float> cyg::pow(const std::vector<float> &v1, const std::vector<float> v2)
{
    auto out_data = new vector<float>(v1.size(), 0);
    op_cpu(*out_data, v1, v2, op_pow);
    return *out_data;
}

std::vector<float> cyg::operator+(const std::vector<float>& v1, const std::vector<float>& v2)
{
    auto out_data = new vector<float>(v1.size(), 0);
    op_cpu(*out_data, v1, v2, op_add);
    return *out_data;
}
std::vector<float> cyg::operator+(const std::vector<float> &v1, const float v2)
{
    auto v2_vec = new vector<float>(v1.size(), v2);
    return v1 + *v2_vec;
}
std::vector<float> cyg::operator-(const std::vector<float> &v1, const std::vector<float> &v2)
{
    auto out_data = new vector<float>(v1.size(), 0);
    op_cpu(*out_data, v1, v2, op_sub);
    return *out_data;
}
std::vector<float> cyg::operator-(const std::vector<float> &v1, const float v2)
{
    auto v2_vec = new vector<float>(v1.size(), v2);
    return v1 - *v2_vec;
}
std::vector<float> cyg::operator*(const std::vector<float> &v1, const std::vector<float> &v2)
{
    auto out_data = new vector<float>(v1.size(), 0);
    op_cpu(*out_data, v1, v2, op_mul);
    return *out_data;
}
std::vector<float> cyg::operator*(const std::vector<float> &v1, const float v2)
{
    auto v2_vec = new vector<float>(v1.size(), v2);
    return v1 * *v2_vec;
}
std::vector<float> cyg::operator/(const std::vector<float> &v1, const std::vector<float> &v2)
{
    auto out_data = new vector<float>(v1.size(), 0);
    op_cpu(*out_data, v1, v2, op_mul);
    return *out_data;
}
std::vector<float> cyg::operator/(const std::vector<float> &v1, const float v2)
{
    auto v2_vec = new vector<float>(v1.size(), v2);
    return v1 / *v2_vec;
}

std::vector<float> cyg::operator/(const float v1, const std::vector<float> &v2)
{
    auto v1_vec = new vector<float>(v2.size(), v1);
    return *v1_vec / v2;
}
/**
 * @brief Save input tensors[shared pointers] that can be retrieved later.
 * the tensors can be retrieved by calling the get_saved_variables() method
 * for ex.
 *  const vector<int>dims{1,2,3};
 *  auto arg1 = ones(dims);
 *  auto arg2 = ones(dims);
 *  ctx->save_for_backward({arg1, arg2});
 *
 * @param tensors(type std::vector<std::shared_ptr<tensor>>)
 */
void cyg::Context::save_for_backward(std::vector<shared_ptr<tensor>> tensors)
{
    for (auto t : tensors)
        this->cache.push_back(t);
}
/**
 * @brief Retrieve tensors that have been saved previously.
 * for ex.
 *  const vector<int>dims{1,2,3};
 *  auto arg1 = ones(dims);
 *  auto arg2 = ones(dims);
 *  ctx->save_for_backward({arg1, arg2});
 *  ctx->get_variables();
 *
 * @return the vectors of tensors shared_pointers (type std::vector<std::shared_ptr<tensor>>)
 */
std::vector<shared_ptr<tensor>> cyg::Context::get_variables() { return this->cache; };

std::shared_ptr<tensor> cyg::Operation::forward(shared_ptr<tensor> lhs, shared_ptr<tensor> rhs)
{
    this->context->save_for_backward({lhs, rhs});
    return std::shared_ptr<tensor>();
}

std::shared_ptr<tensor> cyg::Operation::forward(shared_ptr<tensor> t) { return std::shared_ptr<tensor>(); }

void cyg::Operation::backward(std::vector<float>* incoming_grad) {}

/**
 * @brief operation to run on cpu, plan to use openmpi for some speed ups
 * operation is between two vectors
 * @TODO - consider extending to other types such as int, uint using generics
 * 
 * @param &out(type std::vector<float>) - output parameter
 * @param &lhs(type std::vector<float>) - input parameter - vector 
 * @param &rhs(type std::vector<float>) - input parameter - vector
 * @param &op(type T) - operation to execute, should be a function that accepts two scalars(type: float) and returns a float (same type with output param)
 */
template <typename T>
void cyg::op_cpu(std::vector<float> &out, const std::vector<float> &lhs, const std::vector<float> &rhs, T op)
{
    // #pragma omp parallel for
    for (auto i = 0; i < out.size(); i++)
        out[i] = op(lhs[i], rhs[i]);
};

/**
 * forward and backward passes for basic operations such as Add, etc
 * inherits class Operation
 * it makes use of the context class for caching input and output tensors
 * see the Context class for more information.
 */
// template<typename T>
std::shared_ptr<tensor> cyg::Add::forward(std::shared_ptr<tensor> lhs, std::shared_ptr<tensor> rhs)
{
    assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    vector<float>* out_data;
    if (lhs->get_device() == Device::cpu)
        *out_data = *lhs->data() + *rhs->data();
    // op_cpu(*out_data, *lhs->data(), *rhs->data(), op_add);
    auto out = make_shared<tensor>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    lhs->add_child(out.get());
    rhs->add_child(out.get());
    this->context->save_for_backward({lhs, rhs});

    return out;
}
void cyg::Add::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    string err_msg = "can backprop without a executing a forward computation first";
    assertm(var.size()!=0, err_msg) //prolly not needed though since the lines below wont execute, just for sanity check
    if(var.size()==0) throw runtime_error(err_msg);
    for (auto t : var) if (t->require_grad()) t->update_grad(incoming_grad);
}
std::shared_ptr<tensor> cyg::Mul::forward(std::shared_ptr<tensor> lhs, std::shared_ptr<tensor> rhs)
{
    assert(lhs->get_device() == rhs->get_device() && "tensors are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    vector<float>* out_data;
    if (lhs->get_device() == Device::cpu)
        *out_data = *lhs->data() * *rhs->data();
    // op_cpu(out_data, lhs.data(), rhs.data(), op);
    auto out = make_shared<tensor>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    lhs->add_child(out.get());
    rhs->add_child(out.get());
    this->context->save_for_backward({lhs, rhs});
    return out;
}

void cyg::Mul::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==2, "invalid"); //for debugging purpose
    auto lhs = var[0];
    auto rhs = var[1];
    if(rhs->require_grad()){
        auto grad = *incoming_grad * *lhs->data(); //y=a*b, dy/da = b = 1 * b
        rhs->update_grad(&grad);

    }
    if(lhs->require_grad()){
        auto grad = *incoming_grad * *rhs->data();
        lhs->update_grad(&grad);
    }
}

std::shared_ptr<tensor> cyg::Div::forward(std::shared_ptr<tensor> numerator, std::shared_ptr<tensor> denominator)
{
    assert(numerator->get_device() == denominator->get_device() && "tensors are on different devices");
    if(numerator->get_device()!=denominator->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = numerator->require_grad() || denominator->require_grad();
    vector<float>* out_data;
    if (numerator->get_device() == Device::cpu)
        *out_data = *numerator->data() / *denominator->data();
    // op_cpu(out_data, lhs.data(), rhs.data(), op);
    auto out = make_shared<tensor>(*out_data, numerator->shape(), numerator->get_device(), req_grad);
    numerator->add_child(out.get());
    denominator->add_child(out.get());
    this->context->save_for_backward({numerator, denominator});

    return out;
}

void cyg::Div::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==2, "invalid"); //for debugging purpose
    auto numerator = var[0];
    auto denominator = var[1];
    // y= a/b y = a * b**-1   dy/da = b**-1=1/b  dy/db = a*b**-2
    if(numerator->require_grad()){
        auto local_grad = *incoming_grad * ( 1 / *denominator->data()); //y=a*b, dy/da = b = 1 * b
        numerator->update_grad(&local_grad);
    }
    if(denominator->require_grad()){ //dy/db = a*b**-2
        auto local_grad = *incoming_grad * (*numerator->data() * pow(*denominator->data(), 2));
        denominator->update_grad(&local_grad);
    }
}