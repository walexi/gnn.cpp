#include "operation.h"
#include <memory>
#include <iostream>
using namespace std;
using namespace cyg;

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

std::shared_ptr<tensor> cyg::Operation::forward(const shared_ptr<tensor>& lhs, const shared_ptr<tensor>& rhs)
{
    this->context->save_for_backward({lhs, rhs});
    return std::shared_ptr<tensor>();
}

std::shared_ptr<tensor> cyg::Operation::forward(const shared_ptr<tensor>& t) { return std::shared_ptr<tensor>(); }

void cyg::Operation::backward(std::vector<float>* incoming_grad) {}


/**
 * forward and backward passes for basic operations such as Add, etc
 * inherits class Operation
 * it makes use of the context class for caching input and output tensors
 * see the Context class for more information.
 */
// template<typename T>
std::shared_ptr<tensor> cyg::Add::forward(const std::shared_ptr<tensor>& lhs, const std::shared_ptr<tensor>& rhs)
{
    assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    vector<float>* out_data;
    if (lhs->get_device() == Device::cpu)
        *out_data = *lhs->data() + *rhs->data();
    // op_cpu(*out_data, *lhs->data(), *rhs->data(), op_add);
    auto out = make_shared<tensor>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    out->add_child(lhs.get());
    out->add_child(rhs.get());
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
std::shared_ptr<tensor> cyg::Mul::forward(const std::shared_ptr<tensor>& lhs, const std::shared_ptr<tensor>& rhs)
{
    assert(lhs->get_device() == rhs->get_device() && "tensors are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    vector<float>* out_data;
    if (lhs->get_device() == Device::cpu)
        *out_data = *lhs->data() * *rhs->data();
    // op_cpu(out_data, lhs.data(), rhs.data(), op);
    auto out = make_shared<tensor>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    out->add_child(lhs.get());
    out->add_child(rhs.get());
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

std::shared_ptr<tensor> cyg::Div::forward(const std::shared_ptr<tensor>& numerator, const std::shared_ptr<tensor>& denominator)
{
    assert(numerator->get_device() == denominator->get_device() && "tensors are on different devices");
    if(numerator->get_device()!=denominator->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = numerator->require_grad() || denominator->require_grad();
    vector<float>* out_data;
    if (numerator->get_device() == Device::cpu)
        *out_data = *numerator->data() * *denominator->data();
    // op_cpu(out_data, lhs.data(), rhs.data(), op);
    auto out = make_shared<tensor>(*out_data, numerator->shape(), numerator->get_device(), req_grad);
    out->add_child(numerator.get());
    out->add_child(denominator.get());
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
        auto local_grad = *incoming_grad * cyg::pow(*denominator->data(), -1); //y=a*b, dy/da = b = 1 * b
        numerator->update_grad(&local_grad);
    }
    if(denominator->require_grad()){ //dy/db = a*b**-2
        auto local_grad = *incoming_grad * (*numerator->data() * cyg::pow(*denominator->data(), 2));
        denominator->update_grad(&local_grad);
    }
}