#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "utils.h"
#include "operation.h"
#include "tensor.h"

using namespace std;
using namespace cyg;

const float tol = 1e-40;
const auto compare = [](float a){ return fabs(a)<=tol;};


TEST_CASE("testing context")
{
    auto ctx = make_shared<Context<tensor<float>>>();
    ctx->saved_data["test"] = 3;
    CHECK(ctx->saved_data["n"]!=3);
    CHECK(ctx->saved_data["test"]!=5);
    CHECK(ctx->saved_data["test"]==3);
    vector<size_t>dims = {1,2,3};
    auto arg1 = cyg::randn(dims);
    auto arg2 = cyg::randn(dims);
    ctx->save_for_backward({arg1, arg2});
    auto var = ctx->get_variables();
    vector<shared_ptr<tensor<float>>> initvariables = {arg1, arg2};
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
    auto add = make_shared<Add<tensor<float>>>();
    vector<size_t> dims = {2,3};
    auto lhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto rhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *lhs->data() + *rhs->data();
    auto res = add->forward(lhs, rhs);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 1, Device::cpu, true);
    add->backward(incoming_gradients.get());
    valarray<float> diff2 = *incoming_gradients->data() - *lhs->get_grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));
    add.reset();
    lhs.reset();
    rhs.reset();
    res.reset();
}

TEST_CASE("testing Mul")
{
    auto mul = make_shared<Mul<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto lhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto rhs = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *lhs->data() * *rhs->data();
    auto res = mul->forward(lhs, rhs);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 2, Device::cpu, true);
    mul->backward(incoming_gradients.get());
    auto rhs_grad = *incoming_gradients->data() * *lhs->data();
    valarray<float> diff2 = rhs_grad - *rhs->get_grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));
    auto lhs_grad = *incoming_gradients->data() * *rhs->data();
    valarray<float> diff3 = lhs_grad - *lhs->get_grad();
    CHECK(all_of(begin(diff3), end(diff3), compare));
    mul.reset();
    lhs.reset();
    rhs.reset();
    res.reset();
}

TEST_CASE("testing Div")
{
    auto div = make_shared<Div<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto numerator = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto denominator = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res_vec = *numerator->data() / *denominator->data();
    auto res = div->forward(numerator, denominator);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 2, Device::cpu, true);
    div->backward(incoming_gradients.get());
    auto numerator_grad = *incoming_gradients->data() / *denominator->data(); //y=a/b, dy/da = b**-1 = 1 / b
    valarray<float> diff2 = numerator_grad - *numerator->get_grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));
    auto denominator_grad = (*incoming_gradients->data() * -1) * (*numerator->data() / std::pow(*denominator->data(), 2)); //dy/db = -a*b**-2
    valarray<float> diff3 = denominator_grad - *denominator->get_grad();
    CHECK(all_of(begin(diff3), end(diff3), compare));
    div.reset();
    numerator.reset();
    denominator.reset();
    res.reset();

}

TEST_CASE("testing Pow")
{
    auto pow = make_shared<Pow<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto base = cyg::randn(dims, 3, 7, Device::cpu, true);
    auto exponent = cyg::randn(dims, 1, 4, Device::cpu, true);
    auto res_vec = std::pow(*base->data(), *exponent->data());
    auto res = pow->forward(base, exponent);//prolly not the right way to do it, but i already checked the results with the torch equ
    valarray<float> diff = *res->data() - res_vec;
    CHECK(all_of(begin(diff), end(diff), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 1, Device::cpu, true);
    pow->backward(incoming_gradients.get());
    auto base_grad = *incoming_gradients->data() * *exponent->data() * std::pow(*base->data(), *exponent->data() - 1);      // y = b**e    dy/db = e*b**e-1     
    valarray<float> diff_base_grad = *base->get_grad() - base_grad;
    CHECK(all_of(begin(diff_base_grad), end(diff_base_grad), compare));
    auto exponent_grad = *incoming_gradients->data() * (std::pow(*base->data(), *exponent->data()) * std::log(*base->data())); //dy/db = -a*b**-2
    valarray<float> diff_exponent_grad = exponent_grad - exponent_grad;
    CHECK( all_of(begin(diff_exponent_grad), end(diff_exponent_grad), compare));
    pow.reset();
    base.reset();
    exponent.reset();
    res.reset();
}

// // // @todo check backward passes for the operations below

TEST_CASE("testing Mean")
{
    
    auto m = make_shared<Mean<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto base = cyg::randn(dims, 0, 3, Device::cpu, true);
    
    std::valarray<float> diff;
    
    int d = 1;
    auto res1 = m->forward(base, d);
    // cout<<*res2<<"\n";
    auto res_vec = mean<float>(base->data(), base->shape(), d);//not the way to, already compared with the torch eqv for this operator
    diff = *res1->data() - *res_vec; //expensive, just for testing
    // for(auto d:std::begin(diff))cout<<"d="<<d<<"\t";
    CHECK(all_of(std::begin(diff), std::end(diff), compare));

    d = 1;
    auto res2 = m->forward(base, d);
    res_vec = mean<float>(base->data(), base->shape(), d);//not the way to, already compared with the torch eqv for this operator
    diff = *res2->data() - *res_vec;
    CHECK(all_of(std::begin(diff), std::end(diff), compare));

    d = 0;
    auto res3 = m->forward(base, d);
    res_vec = mean<float>(base->data(), base->shape(), d);//not the way to, already compared with the torch eqv for this operator
    diff = *res3->data() - *res_vec;
    CHECK(all_of(std::begin(diff), std::end(diff), compare));

    m.reset();
    base.reset();

}


TEST_CASE("testing Exp")
{
    auto e = make_shared<Exp<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto base = cyg::randn(dims, -1, 1, Device::cpu, true);
    auto res = e->forward(base);
    auto res_vec = std::exp(*base->data());
    valarray<float> diff = *res->data() - res_vec;
    CHECK(all_of(begin(diff), end(diff), compare));

    res.reset();
    e.reset();
    base.reset();
}

TEST_CASE("testing Log")
{
    auto l= make_shared<Log<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto base = cyg::randn(dims, 1, 4, Device::cpu, true);
    auto res = l->forward(base);
    auto res_vec = std::log(*base->data());
    valarray<float> diff = *res->data() - res_vec;
    CHECK(all_of(begin(diff), end(diff), compare));

    res.reset();
    l.reset();
    base.reset();
}


TEST_CASE("testing Sum")
{
    auto m = make_shared<Sum<tensor<float>>>();
    vector<size_t> dims = {2,3,4};
    auto base = cyg::randn(dims, 0, 3, Device::cpu, true);

    int d = 1;
    valarray<float> res;
    valarray<float> res_vec;
    valarray<float> diff;

    auto res2 = m->forward(base, 1, true);
    auto incoming_gradients = tensor<float>(dims, 1, Device::cpu, true);
    m->backward(&incoming_gradients);
    res2->backward(ones_like(res2).get());
    // for(auto k:*base->get_grad()) cout<<"k="<<k<<"\t";
    m.reset();
    base.reset();

}


TEST_CASE("testing MatMul")
{
    auto mat_op = make_shared<MatMul<tensor<float>>>();
    auto lhs = cyg::randn({2,3,4}, -1, 1, Device::cpu, true);
    auto rhs = cyg::randn({4,3}, -1, 1, Device::cpu, true);
    auto [dims, res_vec] = matmul<tensor<float>, float>(lhs.get(), rhs.get());
    auto res = mat_op->forward(lhs, rhs);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = tensor<float>(dims, 1, Device::cpu, true);
    mat_op->backward(&incoming_gradients);
    auto [_, lhs_grad] =  matmul<tensor<float>, float>(&incoming_gradients, rhs->transpose(-1,-2).get());
    valarray<float> diff2 = std::valarray(lhs_grad[std::slice(0, lhs->n_elements(), 1)]) - *lhs->get_grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));

    auto [__, rhs_grad] =  matmul<tensor<float>, float>(lhs->transpose(-1,-2).get(), &incoming_gradients);
    valarray<float> diff3 = std::valarray(rhs_grad[std::slice(0, rhs->n_elements(), 1)]) - *rhs->get_grad();
    CHECK(all_of(begin(diff3), end(diff3), compare));

    mat_op.reset();
    lhs.reset();
    rhs.reset();
    res.reset();
}