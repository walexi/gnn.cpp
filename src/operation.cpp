#include "operation.h"
#include "tensor.h"
#include "arithmetic.h"
#include <memory>
#include <iostream>
#include <numeric>
// #include <functional>
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
 * @param Ts(type std::vector<std::shared_ptr<T>>)
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
template<>
std::shared_ptr<tensor> cyg::Operation<tensor>::forward(const shared_ptr<tensor>& lhs)
{
    this->context->save_for_backward({lhs});
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
    assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("tensors are on different devices");
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
    for (const auto& t : var) if (t->require_grad()) t->backward(incoming_grad);
    this->context.reset();
}
template<>std::shared_ptr<tensor> cyg::Mul<tensor>::forward(const std::shared_ptr<tensor>& lhs, const std::shared_ptr<tensor>& rhs)
{
    assert(lhs->get_device() == rhs->get_device() && "tensors are on different devices");
    if(lhs->get_device()!=rhs->get_device()) throw runtime_error("tensors are on different devices");
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
    this->context.reset();
}

template<>std::shared_ptr<tensor> cyg::Div<tensor>::forward(const std::shared_ptr<tensor>& numerator, const std::shared_ptr<tensor>& denominator)
{
    assert(numerator->get_device() == denominator->get_device() && "tensors are on different devices");
    if(numerator->get_device()!=denominator->get_device()) throw runtime_error("tensors are on different devices");
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
        denominator->backward(&local_grad);
    }
    this->context.reset();
}

template<>std::shared_ptr<tensor> cyg::Pow<tensor>::forward(const std::shared_ptr<tensor>& base, const std::shared_ptr<tensor>& exponent)
{
    assert(base->get_device() == exponent->get_device() && "tensors are on different devices");
    if(base->get_device()!=exponent->get_device()) throw runtime_error("tensors are on different devices");
    auto req_grad = base->require_grad() || exponent->require_grad();
    auto out_data = new(nothrow) vector<float>();
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    *out_data = cyg::pow(*base->data(), *exponent->data());
    auto out = make_shared<tensor>(*out_data, base->shape(), base->get_device(), req_grad);
    this->context->save_for_backward({base, exponent});

    return out;
}
template<>void cyg::Pow<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==2, "invalid"); //for debugging purpose
    auto base = var[0];
    auto exponent = var[1];
    // y = b**e    dy/db = e*b**e-1     
    if(base->require_grad()){
        auto local_grad = *incoming_grad * (*exponent->data() *  cyg::pow(*base->data(), *exponent->data() - 1));
        base->backward(&local_grad);
    }
    if(exponent->require_grad()){ //logy = elogb dy(1/y) = delogb dy/de = ylogb = b**e * logb     (logb natural log)
        auto local_grad = *incoming_grad * (cyg::pow(*base->data(), *exponent->data()) * cyg::log(*base->data()));
        exponent->backward(&local_grad);
    }
    this->context.reset();    
}

template<>std::shared_ptr<tensor> cyg::Mean<tensor>::forward(const std::shared_ptr<tensor>& base, const int dim, const bool keepdims)
{
    assertm(0<=dim<base->rank(), "out of range");
    if(dim>=base->rank() || dim<0) throw runtime_error("out of bound range");
    vector<int> tdims = base->shape();    
    auto out_data = new(nothrow) vector<float>();
    if(out_data==nullptr) throw runtime_error("insufficient memory");

    *out_data = cyg::mean(*base->data(), tdims, dim);

    tdims.erase(tdims.begin()+dim);
    if(keepdims) tdims.insert(tdims.begin()+dim, 1);

    auto out = make_shared<tensor>(*out_data, tdims, base->get_device(), base->require_grad());
    this->context->save_for_backward({base});
    this->context->saved_data["dims"] = dim;
    this->context->saved_data["keepdims"] = keepdims;

    return out;
}

template<>void cyg::Mean<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==1, "invalid"); //for debugging purpose
    assertm(this->context->saved_data.size()==2, "invalid");
    auto dims = this->context->saved_data["dims"];
    auto keepdims = this->context->saved_data["keepdims"];
    auto base = var[0];
    // y = (a_1 + a_2 + ... + a_n) / n    where n = base->shape()[dims]
    // dy/da_1 = 1/n
    auto out_data = new(nothrow) vector<float>(base->n_elements(), 1/base->shape()[dims]);
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    if(base->require_grad()){
        auto local_grad = *incoming_grad * *out_data;
        base->backward(&local_grad);
    }
    this->context.reset();
}

template<>std::shared_ptr<tensor> cyg::Exp<tensor>::forward(const std::shared_ptr<tensor>& t)
{
    auto out_data = new(nothrow) vector<float>();
    if(out_data==nullptr) throw runtime_error("insufficient memory");

    *out_data = cyg::exp(*t->data());

    auto out = make_shared<tensor>(*out_data, t->shape(), t->get_device(), t->require_grad());
    this->context->save_for_backward({t});

    return out;
}

template<>void cyg::Exp<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==1, "invalid"); //for debugging purpose
    // y= e**x   logy = xloge   logy = x  i/y(dy) = dx  dy/dx = y = e**x 
    auto base = var[0];
    auto out_data = new(nothrow) vector<float>(base->n_elements(), 0);
    *out_data = cyg::exp(*base->data());
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    if(base->require_grad()){
        auto local_grad = *incoming_grad * *out_data;
        base->backward(&local_grad);
    }
    this->context.reset();
}

template<>std::shared_ptr<tensor> cyg::Log<tensor>::forward(const std::shared_ptr<tensor>& t, const char& base)
{
    // assertm(dims<3 && dims>-1, "current implementation only works for dims in the range [0,2]");
    auto out_data = new(nothrow) vector<float>();
    if(out_data==nullptr) throw runtime_error("insufficient memory");

    *out_data = cyg::log(*t->data());

    auto out = make_shared<tensor>(*out_data, t->shape(), t->get_device(), t->require_grad());
    this->context->save_for_backward({t});
    // this->context->saved_data["base"] = base;

    return out;
}

template<>void cyg::Log<tensor>::backward(std::vector<float>* incoming_grad)
{
    auto var = this->context->get_variables();
    assertm(var.size()==1, "invalid"); //for debugging purpose
    // y= e**x   logy = xloge   logy = x  i/y(dy) = dx  dy/dx = y = e**x 
    auto base = var[0];
    auto out_data = new(nothrow) vector<float>(base->n_elements(),0);
    if(out_data==nullptr) throw runtime_error("insufficient memory");
    *out_data = cyg::pow(*base->data(),-1);
    if(base->require_grad()){
        auto local_grad = *incoming_grad * *out_data;
        base->backward(&local_grad);
    }
    this->context.reset();
}