#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "operation.h"
#include "tensor.h"
#include <memory>

using namespace std;
using namespace cyg;


TEST_CASE("testing context")
{
    auto ctx = make_shared<Context>();
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
    auto add = make_shared<Add>();
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
    auto mul = make_shared<Mul>();
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
    auto div = make_shared<Div>();
    vector<int> dims = {2,3,4};
    auto numerator = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto denominator = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *numerator->data() / *denominator->data();
    auto res = div->forward(numerator, denominator);
    CHECK(equal(res->data()->cbegin(), res->data()->cend(), res_vec.begin()));
    auto incoming_gradients = initialize(dims, 2);
    div->backward(incoming_gradients);
    auto numerator_grad = *incoming_gradients * ( 1/ *denominator->data() ); //y=a*b, dy/da = b = 1 * b
    CHECK(equal(numerator_grad.cbegin(), numerator_grad.cend(), numerator->get_grad()->cbegin()));
    auto denominator_grad = *incoming_gradients * ( *numerator->data() * cyg::pow(*denominator->data(), 2) ); //dy/db = a*b**-2
    CHECK(equal(denominator_grad.cbegin(), denominator_grad.cend(), denominator->get_grad()->cbegin()));
    div.reset();
    numerator.reset();
    denominator.reset();
    res.reset();

}