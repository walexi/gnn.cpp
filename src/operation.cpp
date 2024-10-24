#include "operation.h"
#include "tensor.h"
#include "arithmetic.h"
#include <memory>
#include <iostream>
using namespace std;
using namespace cyg;

/**
 * @brief Save input Ts[shared pointers] that can be retrieved later.
 * the Ts can be retrieved by calling the get_saved_variables() method
 * for ex.
 *  const vector<int>dims{1,2,3};
 *  auto arg1 = ones(dims);
 *  auto arg2 = ones(dims);
 *  ctx->save_for_backward({arg1, arg2});
 *
 * @param Ts(type std::vector<std::shared_ptr<tensor>>)
 */
template<>
void cyg::Context<tensor>::save_for_backward(std::vector<shared_ptr<tensor>> Ts)
{
    for (auto t : Ts)
        this->cache.push_back(t);
}
/**
 * @brief Retrieve Ts that have been saved previously.
 * for ex.
 *  const vector<int>dims{1,2,3};
 *  auto arg1 = ones(dims);
 *  auto arg2 = ones(dims);
 *  ctx->save_for_backward({arg1, arg2});
 *  ctx->get_variables();
 *
 * @return the vectors of Ts shared_pointers (type std::vector<std::shared_ptr<tensor>>)
 */
template<>
std::vector<shared_ptr<tensor>> cyg::Context<tensor>::get_variables() { return this->cache; };

template<>
std::shared_ptr<tensor> cyg::Operation<tensor>::forward(const shared_ptr<tensor>& lhs, const shared_ptr<tensor>& rhs)
{
    this->context->save_for_backward({lhs, rhs});
    return std::shared_ptr<tensor>();
}
template<>void cyg::Operation<tensor>::backward(std::vector<float>* incoming_grad) {}


/**
 * forward and backward passes for basic operations such as Add, etc
 * inherits class Operation
 * it makes use of the Context<tensor> class for caching input Ts for backward computation
 * see the Context<tensor> class for more information.
 */
template<>std::shared_ptr<tensor> cyg::Add<tensor>::forward(const std::shared_ptr<tensor>& lhs, const std::shared_ptr<tensor>& rhs)
{
    assertm(lhs->get_device() == rhs->get_device(), "Ts are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("Ts are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    auto out_data = new(nothrow) vector<float>();// allocating object on the heap,dont forget
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    *out_data = *lhs->data() + *rhs->data();
    auto out = make_shared<tensor>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    this->context->save_for_backward({lhs, rhs});

    return out;
}
template<>void cyg::Add<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    string err_msg = "can backprop without a executing a forward computation first";
    // assertm(var.size()!=0, err_msg); //prolly not needed though since the lines below wont execute, just for sanity check
    if(var.size()==0) throw runtime_error(err_msg);
    for (auto t : var) if (t->require_grad()) t->backward(incoming_grad);
    // this->Context<tensor>.reset();
}
template<>std::shared_ptr<tensor> cyg::Mul<tensor>::forward(const std::shared_ptr<tensor>& lhs, const std::shared_ptr<tensor>& rhs)
{
    assert(lhs->get_device() == rhs->get_device() && "Ts are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("Ts are on different devices");
    auto req_grad = lhs->require_grad() || rhs->require_grad();
    auto out_data = new(nothrow) vector<float>();
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    *out_data = *lhs->data() * *rhs->data();
    auto out = make_shared<tensor>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
    this->context->save_for_backward({lhs, rhs});
    return out;
}

template<>void cyg::Mul<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    // assertm(var.size()==2, "invalid"); //for debugging purpose
    auto lhs = var[0];
    auto rhs = var[1];
    if(rhs->require_grad()){
        auto grad = *incoming_grad * *lhs->data(); //y=a*b, dy/da = b = 1 * b
        rhs->backward(&grad);
    }
    if(lhs->require_grad()){
        auto grad = *incoming_grad * *rhs->data();
        lhs->backward(&grad);
    }
}

template<>std::shared_ptr<tensor> cyg::Div<tensor>::forward(const std::shared_ptr<tensor>& numerator, const std::shared_ptr<tensor>& denominator)
{
    assert(numerator->get_device() == denominator->get_device() && "Ts are on different devices");
    if(numerator->get_device()!=denominator->get_device()) throw runtime_error("Ts are on different devices");
    auto req_grad = numerator->require_grad() || denominator->require_grad();
    auto out_data = new(nothrow) vector<float>();
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    *out_data = *numerator->data() / *denominator->data();
    auto out = make_shared<tensor>(*out_data, numerator->shape(), numerator->get_device(), req_grad);
    this->context->save_for_backward({numerator, denominator});

    return out;
}

template<>void cyg::Div<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==2, "invalid"); //for debugging purpose
    auto numerator = var[0];
    auto denominator = var[1];
    // y= a/b y = a * b**-1   dy/da = b**-1=1/b  dy/db = -a*b**-2
    if(numerator->require_grad()){
        auto local_grad = *incoming_grad * cyg::pow(*denominator->data(), -1);
        numerator->backward(&local_grad);
    }
    if(denominator->require_grad()){ //dy/db = -a*b**-2
        auto local_grad = (*incoming_grad * -1) * (*numerator->data() / cyg::pow(*denominator->data(), 2));
        cout<<printND(local_grad, numerator->shape()).str()<<endl;
        denominator->backward(&local_grad);
    }
}