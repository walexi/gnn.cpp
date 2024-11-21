#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tensor.h"
#include "utils.h"
#include "operation.h"

using namespace std;
using namespace cyg;

const float tol = 1e-20;
const auto compare = [](float a)
{ return fabs(a) <= tol; };

TEST_CASE("testing context")
{
    auto ctx = make_shared<Context<cyg::tensor<float>>>();
    ctx->saved_data["test"] = 3;
    CHECK(ctx->saved_data["n"] != 3);
    CHECK(ctx->saved_data["test"] != 5);
    CHECK(ctx->saved_data["test"] == 3);
    vector<size_t> dims = {1, 2, 3};
    auto arg1 = cyg::randn(dims);
    auto arg2 = cyg::randn(dims);
    ctx->save_for_backward({arg1, arg2});
    auto var = ctx->get_variables();
    vector<shared_ptr<tensor<float>>> initvariables = {arg1, arg2};
    CHECK(equal(var.cbegin(), var.cend(), initvariables.cbegin()));
    CHECK(initvariables.size() == var.size());
    CHECK(initvariables[0] == var[0]);
};

TEST_CASE("testing Add")
{
    auto add = make_shared<Add<cyg::tensor<float>>>();
    vector<size_t> dims = {2, 5};
    auto lhs = cyg::randn(dims, -1, 1, true);
    auto rhs = cyg::randn(dims, -1, 1, true);
    auto res_vec = *lhs->data() + *rhs->data();
    auto res = add->forward(lhs, rhs);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = cyg::randn(res->shape());
    add->backward(incoming_gradients);
    valarray<float> diff2 = *incoming_gradients->data() - *lhs->grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));
}

TEST_CASE("testing Mul")
{
    auto mul = make_shared<Mul<tensor<float>>>();
    vector<size_t> dims = {2, 3, 4};
    auto lhs = cyg::randn(dims, -1, 1, true);
    auto rhs = cyg::randn(dims, -1, 1, true);
    auto res_vec = *lhs->data() * *rhs->data();
    auto res = mul->forward(lhs, rhs);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 2, false);
    mul->backward(incoming_gradients);
    auto rhs_grad = *incoming_gradients->data() * *lhs->data();
    valarray<float> diff2 = rhs_grad - *rhs->grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));
    auto lhs_grad = *incoming_gradients->data() * *rhs->data();
    valarray<float> diff3 = lhs_grad - *lhs->grad();
    CHECK(all_of(begin(diff3), end(diff3), compare));
}

TEST_CASE("testing Div")
{
    auto div = make_shared<Div<tensor<float>>>();
    vector<size_t> dims = {2, 3, 4};
    auto numerator = cyg::randn(dims, -1, 1, true);
    auto denominator = cyg::randn(dims, -1, 1, true);
    auto res_vec = *numerator->data() / *denominator->data();
    auto res = div->forward(numerator, denominator);
    valarray<float> diff1 = *res->data() - res_vec;
    CHECK(all_of(begin(diff1), end(diff1), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 2, false);
    div->backward(incoming_gradients);
    auto numerator_grad = *incoming_gradients->data() / *denominator->data(); // y=a/b, dy/da = b**-1 = 1 / b
    valarray<float> diff2 = numerator_grad - *numerator->grad();
    CHECK(all_of(begin(diff2), end(diff2), compare));
    auto denominator_grad = (*incoming_gradients->data() * -1) * (*numerator->data() / std::pow(*denominator->data(), 2)); // dy/db = -a*b**-2
    valarray<float> diff3 = denominator_grad - *denominator->grad();
    CHECK(all_of(begin(diff3), end(diff3), compare));
}

TEST_CASE("testing Pow")
{
    auto pow = make_shared<Pow<tensor<float>>>();
    vector<size_t> dims = {2, 3, 1};
    auto base = cyg::randn(dims, 3, 7, true);
    auto exponent = cyg::randn(dims, 1, 4, true);
    auto res_vec = std::pow(*base->data(), *exponent->data());
    auto res = pow->forward(base, exponent); // prolly not the right way to do it, but i already checked the results with the torch equ
    valarray<float> diff = *res->data() - res_vec;
    CHECK(all_of(begin(diff), end(diff), compare));
    auto incoming_gradients = make_shared<tensor<float>>(dims, 1, false);
    pow->backward(incoming_gradients);
    auto base_grad = *incoming_gradients->data() * *exponent->data() * std::pow(*base->data(), *exponent->data() - 1); // y = b**e    dy/db = e*b**e-1
    valarray<float> diff_base_grad = *base->grad() - base_grad;
    CHECK(all_of(begin(diff_base_grad), end(diff_base_grad), compare));
    auto exponent_grad = *incoming_gradients->data() * (std::pow(*base->data(), *exponent->data()) * std::log(*base->data())); // dy/db = -a*b**-2
    valarray<float> diff_exponent_grad = exponent_grad - exponent_grad;
    CHECK(all_of(begin(diff_exponent_grad), end(diff_exponent_grad), compare));
}

// // // // // @todo check backward passes for the operations below

TEST_CASE("testing Mean")
{

    auto m = make_shared<Mean<tensor<float>>>();
    vector<size_t> dims = {2, 3, 1, 4};
    auto base = cyg::randn(dims, 1, 3, true);

    int d = 0;
    auto res1 = m->forward(base, d, false);
    auto res_t = functional::sum(base, d);
    std::valarray<float> diff = *res1->data() - (*(res_t / base->shape()[d])->data());
    // m->backward(cyg::randn(res1->shape()));
    CHECK(all_of(std::begin(diff), std::end(diff), compare));

    d = 1;
    auto res2 = m->forward(base, d, false);
    auto res_t2 = functional::sum(base, d);
    std::valarray<float> diff2 = *res2->data() - (*(res_t2 / base->shape()[d])->data());
    CHECK(all_of(std::begin(diff2), std::end(diff2), compare));

    d = INT_MAX;
    auto res3 = m->forward(base, d);
    auto res_t3 = functional::sum(base, d);
    diff = *res3->data() - (*(res_t3 / base->numel())->data());
    CHECK(all_of(std::begin(diff), std::end(diff), compare));
}

TEST_CASE("testing Exp")
{
    auto e = make_shared<Exp<tensor<float>>>();
    vector<size_t> dims = {2, 3, 4};
    auto base = cyg::randn(dims, -1, 1, true);
    auto res = e->forward(base);
    auto res_vec = std::exp(*base->data());
    valarray<float> diff = *res->data() - res_vec;
    CHECK(all_of(begin(diff), end(diff), compare));
}

TEST_CASE("testing Log")
{
    auto l = make_shared<Log<tensor<float>>>();
    vector<size_t> dims = {2, 3, 4};
    auto base = cyg::randn(dims, 1, 4, true);
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
    vector<size_t> dims = {2, 3, 4};
    auto base = cyg::randn(dims, 0, 3, true);

    int d = INT_MAX;
    valarray<float> res;
    valarray<float> res_vec;
    valarray<float> diff;

    auto res2 = m->forward(base, d, true);
    auto incoming_gradients = make_shared<tensor<float>>(dims, 1, false);
    m->backward(incoming_gradients);
    m.reset();
    base.reset();
}

TEST_CASE("testing Var")
{
    auto var_op = make_shared<Var<tensor<float>>>();

    auto data = new std::valarray<float>();
    *data = {0.4010f, -0.1711f, 0.6153f, 0.3418f, 0.8220f, 0.5479f, 0.0709f, 0.7616f, 0.9428f, -0.8060f, -0.7953f, -0.1762f, -0.2522f, 0.0513f, -0.0275f, 0.6111f, 0.3995f, -0.9120f, -0.8934f, -0.8242f, -0.4318f, -0.8992f, -0.9997f, -0.5030f};
    std::vector<size_t> dims = {2, 3, 4};
    auto base = std::make_shared<tensor<float>>(dims, data, true);
    int d = INT_MAX;
    auto res = var_op->forward(base, d, true);
    var_op->backward(make_shared<tensor<float>>(res->shape(), 1, false));

    std::valarray<float> res_vec = {
        0.0426f, -0.0072f, 0.0612f, 0.0374f,
        0.0792f, 0.0553f, 0.0139f, 0.0739f,
        0.0897f, -0.0624f, -0.0615f, -0.0076f,
        -0.0142f, 0.0122f, 0.0053f, 0.0608f,
        0.0424f, -0.0716f, -0.0700f, -0.0640f,
        -0.0298f, -0.0705f, -0.0792f, -0.0360f};
    std::valarray<float> diff = std::abs(*base->grad() - res_vec);

    CHECK(all_of(std::begin(diff), std::end(diff), compare));
    // std::cout<<printND(&diff, base->shape()).str()<<"\n";

    base->zero_grad();

    d = -1;
    auto res2 = var_op->forward(base, d, true);
    var_op->backward(make_shared<tensor<float>>(res->shape(), 1, false));

    std::valarray<float> res_vec2 = {0.0695, -0.3119, 0.2124, 0.0300, 0.1809, -0.0018, -0.3198, 0.1407, 0.7676, -0.3982, -0.3911, 0.0217, -0.2319, -0.0296, -0.0821, 0.3436, 0.6380, -0.2363, -0.2239, -0.1778, 0.1844, -0.1272, -0.1942, 0.1370};
    std::valarray<float> diff2 = std::abs(*base->grad() - res_vec2);

    CHECK(all_of(std::begin(diff2), std::end(diff2), compare));
    var_op.reset();
    base.reset();
}

TEST_CASE("testing MatMul")
{
    auto mat_op = make_shared<MatMul<tensor<float>>>();
    auto lhs = cyg::randn({1, 2, 3, 4}, -1, 1, true);
    auto rhs = cyg::randn({2, 4, 3}, -1, 1, true);
    auto out = functional::matmul(lhs, rhs);
    auto res = mat_op->forward(lhs, rhs);

    auto incoming_gradients = cyg::randn(res->shape());
    mat_op->backward(incoming_gradients);
    // auto lhs_grad =  matmul<tensor<float>>(incoming_gradients, rhs->transpose(-1,-2));

    mat_op.reset();
    lhs.reset();
    rhs.reset();
    res.reset();
}
