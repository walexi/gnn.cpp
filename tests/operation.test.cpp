#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "operation.h"
#include "tensor.h"

using namespace std;
using namespace cyg;

const float tol = 1e-20;
const auto compare = [](float a){ return fabs(a)<=tol;};


TEST_CASE("testing context")
{
    auto ctx = make_shared<Context<tensor>>();
    ctx->saved_data["test"] = 3;
    CHECK(ctx->saved_data["n"]!=3);
    CHECK(ctx->saved_data["test"]!=5);
    CHECK(ctx->saved_data["test"]==3);
    vector<int>dims = {1,2,3};
    auto arg1 = cyg::randn(dims);
    auto arg2 = cyg::randn(dims);
    ctx->save_for_backward({arg1, arg2});
    auto var = ctx->get_variables();
    vector<shared_ptr<tensor>> initvariables = {arg1, arg2};
    CHECK(equal(var.cbegin(), var.cend(), initvariables.cbegin()));
    CHECK(initvariables.size() == var.size());
    CHECK(initvariables[0] == var[0]);
    arg1.reset();
    arg2.reset();
    ctx.reset();
    initvariables.clear();
};


TEST_CASE("testing Add")
{
    auto add = make_shared<Add<tensor>>();
    vector<int> dims = {2,3};
    auto lhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto rhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *lhs->data() + *rhs->data();
    auto res = add->forward(lhs, rhs);
    CHECK(equal(res->data()->cbegin(), res->data()->cend(), res_vec.begin()));
    auto incoming_gradients = initialize(dims, 1);
    add->backward(incoming_gradients);
    CHECK(equal(incoming_gradients->cbegin(), incoming_gradients->cend(), lhs->get_grad()->cbegin())); //since lhs grad were prev zero, it should now be equal to incoming grad
    add.reset();
    lhs.reset();
    rhs.reset();
    res.reset();
}

TEST_CASE("testing Mul")
{
    auto mul = make_shared<Mul<tensor>>();
    vector<int> dims = {2,3,4};
    auto lhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto rhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *lhs->data() * *rhs->data();
    auto res = mul->forward(lhs, rhs);
    CHECK(equal(res->data()->cbegin(), res->data()->cend(), res_vec.begin()));
    auto incoming_gradients = initialize(dims, 2);
    mul->backward(incoming_gradients);
    auto lhs_grad = *incoming_gradients * *rhs->data();
    CHECK(equal(lhs_grad.cbegin(), lhs_grad.cend(), lhs->get_grad()->cbegin()));
    auto rhs_grad = *incoming_gradients * *lhs->data();
    CHECK(equal(rhs_grad.cbegin(), rhs_grad.cend(), rhs->get_grad()->cbegin()));
    mul.reset();
    lhs.reset();
    rhs.reset();
    res.reset();
}

TEST_CASE("testing Div")
{
    auto div = make_shared<Div<tensor>>();
    vector<int> dims = {2,3,4};
    auto numerator = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto denominator = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *numerator->data() / *denominator->data();
    auto res = div->forward(numerator, denominator);
    CHECK(equal(res->data()->cbegin(), res->data()->cend(), res_vec.begin()));
    auto incoming_gradients = initialize(dims, 2);
    div->backward(incoming_gradients);
    auto numerator_grad = *incoming_gradients * cyg::pow(*denominator->data(),-1); //y=a*b, dy/da = b = 1 * b
    CHECK(equal(numerator_grad.cbegin(), numerator_grad.cend(), numerator->get_grad()->cbegin()));
    auto denominator_grad = (*incoming_gradients * -1) * (*numerator->data() / cyg::pow(*denominator->data(), 2)); //dy/db = -a*b**-2
    CHECK(equal(denominator_grad.cbegin(), denominator_grad.cend(), denominator->get_grad()->cbegin()));
    div.reset();
    numerator.reset();
    denominator.reset();
    res.reset();

}



TEST_CASE("testing Pow")
{
    auto pow = make_shared<Pow<tensor>>();
    vector<int> dims = {2,3,4};
    auto base = cyg::randn(dims, 3, 7, Device::cpu, true);
    auto exponent = cyg::fill(5, dims, Device::cpu, true);
    auto res_vec = cyg::pow(*base->data(), *exponent->data());
    auto res = pow->forward(base, exponent);//prolly not the right way to do it, but i already checked the results with the torch equ
    auto diff = *res->data() - res_vec;
    CHECK(all_of(diff.begin(), diff.end(), compare));
    auto incoming_gradients = initialize(dims, 1);
    pow->backward(incoming_gradients);
    auto base_grad = *incoming_gradients * *exponent->data() * cyg::pow(*base->data(), *exponent->data() - 1);      // y = b**e    dy/db = e*b**e-1     
    auto diff_base_grad = *base->get_grad() - base_grad;
    CHECK( all_of(diff_base_grad.begin(), diff_base_grad.end(), compare));
    auto exponent_grad = *incoming_gradients * (cyg::pow(*base->data(), *exponent->data()) * cyg::log(*base->data())); //dy/db = -a*b**-2
    auto diff_exponent_grad = exponent_grad - exponent_grad;
    CHECK( all_of(diff_exponent_grad.begin(), diff_exponent_grad.end(), compare));
    pow.reset();
    base.reset();
    exponent.reset();
    res.reset();
}

// @todo check backward passes for the operations below

TEST_CASE("testing Mean")
{
    
    auto m = make_shared<Mean<tensor>>();
    vector<int> dims = {2,3,4};
    auto base = cyg::randn(dims, -1, 1, Device::cpu, true);
    CHECK_THROWS(m->forward(base, 5));

    int d = 2;
    vector<float> res;
    vector<float> res_vec;
    vector<float> diff;
 
    res = (*m->forward(base, d)->data());
    res_vec = cyg::mean(*base->data(), base->shape(), d);//not the way to, already compared with the torch eqv for this operator
    diff = res - res_vec;
    CHECK(all_of(diff.cbegin(), diff.cend(), compare));

    d = 1;
    res = (*m->forward(base, d)->data());
    res_vec = cyg::mean(*base->data(), base->shape(), d);//not the way to, already compared with the torch eqv for this operator
    diff = res - res_vec;
    CHECK(all_of(diff.cbegin(), diff.cend(), compare));

    d = 0;
    res = (*m->forward(base, d)->data());
    res_vec = cyg::mean(*base->data(), base->shape(),d);//not the way to, already compared with the torch eqv for this operator
    diff = res - res_vec;
    CHECK(all_of(diff.begin(), diff.end(), compare));
    m.reset();
    base.reset();

}

TEST_CASE("testing Exp")
{
    auto e = make_shared<Exp<tensor>>();
    vector<int> dims = {2,3,4};
    auto base = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res = e->forward(base);
    auto res_vec = cyg::exp(*base->data());
    auto diff = *res->data() - res_vec;
    CHECK(all_of(diff.begin(), diff.end(), compare));

    res.reset();
    e.reset();
    base.reset();
}

TEST_CASE("testing Log")
{
    auto l= make_shared<Log<tensor>>();
    vector<int> dims = {2,3,4};
    auto base = cyg::randn(dims, 1, 4, Device::cpu, true);
    auto res = l->forward(base);
    auto res_vec = cyg::log(*base->data());
    auto diff = *res->data() - res_vec;
    CHECK(all_of(diff.begin(), diff.end(), compare));

    res.reset();
    l.reset();
    base.reset();

}