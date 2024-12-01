#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn.h"
#include "tensor.h"
#include "operation.h"
#include "utility"
#include "functional.h"
#include <memory>
#include <vector>

using namespace std;
using namespace nn;
using namespace cyg;

const float tol = 1e-40;
const auto compare = [](float a){ return fabs(a)<=tol;};


TEST_CASE("testing module")
{
    auto module = make_shared<Module>();
    vector<size_t> dims = {2,3,4};
    shared_ptr<tensor<float>> param1 = make_shared<tensor<float>>(dims, 1, true);
    module->register_parameter("m1", param1);
    module->eval();
    CHECK(module->training==false);
    module->train();
    CHECK(module->training==true);
    CHECK(module->parameters().size()==1);

};

TEST_CASE("testing linear")
{

    // auto out_tensor = (*lln)(inp_tensor);
    // std::cout<<*out_tensor<<"\n";
    // auto out_idx = out_tensor->argmax(-1, true);

    // std::cout<<*out_idx<<"\n";

    // // auto labels = new std::valarray<int>(batch * samples);
    // // std::iota(std::begin(*labels), std::end(*labels), 0);

    // // auto labels_tensor = cyg::ones_like(out_idx);
    // // labels_tensor->set_data(labels);

    // out_tensor->backward(cyg::randn(out_tensor->shape()));
    // // auto acc = functional::abs(out_idx - labels_tensor)->to_float(); //@todo work on explicit conversion (tensor<int> to tensor<float>)
    // // auto acc_ = acc->mean(); // @todo mean op return floating values
    // // // std::cout<<*acc_<<"\n";
    // std::cout<<printND(lln->weight->grad(), lln->weight->shape()).str()<<"\n";

    // // auto m_v = cyg::randn({4,1,5})->gt(cyg::randn({4,5}));

    // auto relu = nn::ReLU();
    // auto data = new std::valarray<float>();
    // *data = {-1.8815,   1.0678,  -2.3728,  -2.6861,   0.6901, 1.7538,   1.6826,  -0.7102,  -2.7788,  -1.4798 , -1.8314,   0.6989,  -2.2111,   2.9825,  -2.8285, -2.4978,  -0.1588,   2.1702,  -1.2652,   2.2216, -0.4209,  -2.4894,   2.3382,  -0.2480,  -2.7402  };
    // std::vector<size_t> dims = {5, 5};
    // auto t2 = std::make_shared<tensor<float>>(dims, data, true);
    
    // auto t2 = cyg::randn({10, 15}, -3, 3, true);
    // auto l = t2->where(t2>0.0f, 0.0f);
    // auto abc = relu(inp_tensor);
    // // // // abc->to_bool();
    // l->backward(cyg::ones_like(l)->to_float());
    // std::cout<<*t2<<"\n";


    // auto dropout = nn::Dropout(0.4);
    // auto out = dropout(t2);

    // out->backward(cyg::randn(out->shape()));
    // std::cout<<*t2<<"\n";

    // auto softmax = nn::Softmax(-1);
    // auto out = softmax(t2);

    // // auto bnorm = nn::BatchNorm(15);
    // // auto out = bnorm(t2);

    // auto lnorm = nn::LayerNorm(15);
    // std::cout<<*out<<"\n";
    const size_t h_in = 3;
    const size_t h_out = 5;
    const size_t h_cell = 5;

    size_t samples = 10;
    size_t batch = 5;
    size_t n_layers = 5;
    auto inp_tensor = cyg::randn({samples, h_in}, -2, 2, true);
    auto h0 = cyg::randn({n_layers, batch, h_out}, -2, 2, true);
    auto c0 = cyg::randn({n_layers, batch, h_cell}, -2, 2, true);

    auto m = Sequential({ {"lin1", new Linear(h_in, h_out, true)},{"lin2", new Linear(h_out, h_out, true)}});
    // // // auto seq_model2 = Sequential();
    // // // seq_model2.register_module("tre", new Linear(in_features, out_features));
    // // std::cout<<"size="<<m.parameters().size()<<"\n";
    // // auto opt = SGD(m.parameters(), 0.5);

    auto y = m(inp_tensor);
    std::cout<<printND(inp_tensor->grad(), inp_tensor->shape()).str()<<"\n";
    // auto dimm = {samples};
    // auto target = make_shared<tensor<int>>(dimm, 0, false);
    
    // // std::cout<<*target<<"\n";
    std::cout<<*y<<"\n";
    // auto closs = cross_entropy_loss(y, target);
    // closs->backward();
    // std::cout<<*closs<<"\n";
    // std::cout<<printND(y->grad(), y->shape()).str()<<"\n";
    auto inp = cyg::randn({3, 3});
    inp->fill_diagonal_(2);
    std:cout<<*inp<<"\n";
    std::cout<<*eye(2)<<"\n";
    // // // auto lstm = LSTM_Layer(h_in, h_out, true);
    // // // auto out = lstm.forward({inp_tensor, h0, c0});
    // // // std::cout<<*std::get<0>(out)<<"\n";
    // // // std::cout<<*out->sum(-1)<<"\n";
    // // // auto out = lnorm(t2);
    
    // // // out->backward(make_shared<tensor<float>>(out->shape(), 1, false));

    // // // std::cout<<*y<<"\n";
    // // std::cout<<printND(inp_tensor->grad(), inp_tensor->shape()).str()<<"\n";
    // // // std::cout<<seq_model<<"\n";
    // // // auto inp_tensor2 = cyg::randn( {1});
    // std::cout<<*m.get_module("lin1")->get_parameter("weight");
    // opt.step();
    // opt.step();
    // std::cout<<*m.get_module("lin1")->get_parameter("weight");
    // cout<<*inp_tensor<<endl;
    // cout<<*inp_tensor2<<endl;

    // lln->zero_grad();
    // // cout<<printND(*inp_tensor->get_grad(), inp_tensor->shape()).str()<<"\n";
    // // cout<<*inp_tensor<<"\n";

    // // inp_tensor->repeat(0, 4);
    // out->backward(ones_like(out).get());
    // cout<<printND(*inp_tensor->get_grad(), inp_tensor->shape()).str()<<"\n";
    // auto g = *lln->parameters()["weight"]->get_grad();
    // cout<<printND(g, lln->parameters()["weight"]->shape()).str()<<"\t";
    // cout<<*out<<"\n";
    // cout<<*out->var(2)<<"\n";
}