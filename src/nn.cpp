#include "nn.h"
#include <memory>
#include <vector>
#include "operation.h"

using namespace nn;
using namespace std;
using namespace cyg;

// void nn::Module::add_params(vector<pair<string, shared_ptr<tensor<float>>>> input_params)
// {
//     for (auto p : input_params) params->insert(p);
// }
// void nn::Module::zero_grad()
// {
//     for (auto &p : *params)  p.second->zero_grad();
// }

// nn::Linear::Linear(int in_features, int out_features, bool bias, Device device): Module()
// {
//     vector<int> dims = {out_features, in_features};
//     auto w = make_shared<tensor<float>>(dims, device);
//     if (bias)
//     {
//         dims = {1, out_features};
//         auto b = make_shared<tensor<float>>(dims, device);
//         this->add_params({make_tuple("weight", w), make_tuple("bias", b)});
//     }
//     else
//         this->add_params({make_tuple("weight", w)});
// }
// std::shared_ptr<cyg::tensor<float>> nn::Linear::forward(const std::shared_ptr<cyg::tensor<float>> &input_tensor)
// {
//     auto w = (*this->params)["weight"];
//     auto b = (*this->params)["bias"];
//     auto output = w->mm(input_tensor)->add(b);

//     return output;
// };

// nn::Sequential::Sequential(vector<std::shared_ptr<Module>> input_modules)
// {
//     int i = 0;
//     for (const auto &m : input_modules)
//         (*this->modules)[to_string(++i)] = m;
// };

// template <class T>
// inline std::shared_ptr<cyg::tensor> nn::Sequential::operator()(const T &input_tensor)
// {
//     A *input_t;
//     *input_t = &input_tensor;
//     for (int i = 0; i < this->modules->size(); i++)
//     {
//         *input_t = this->modules[i](*input_t)
//     }
//     return *input_t;
// };

// inline void Sequential::add_module(string name, std::shared_ptr<Module> m)
// {
//     (*this->modules)[name] = m;
// };


// // https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
// template <>
// std::shared_ptr<tensor<float>> cyg::Dropout<tensor<float>>::forward(const std::shared_ptr<tensor<float>> &t)
// {

//     auto out_data = new (nothrow) valarray<float>();
//     if (out_data == nullptr)
//         throw runtime_error("insufficient memory");
//     bool req_grad;
//     if (!this->isTrain)
//     {
//         // at test time, scale the values by p
//         static auto f = this->p;
//         req_grad = false; // turn of grad for out
//         *out_data = (*t->data()).apply([](float k) -> float
//                                        { return float(k / f); });
//     }
//     else
//     {
//         std::bernoulli_distribution d(1 - this->p);
//         auto mask = new valarray<float>(t->n_elements());
//         generate(begin(*mask), end(*mask), [=, &d]()
//                  { return float(d(re)); });
//         auto mask_tensor = make_shared<tensor<float>>(*mask, t->shape(), t->get_device(), false); // create tensor so i can cache for backward pass
//         *out_data = 0.0f + *mask * *t->data() / (1 - this->p);                                    // adding 0.0f as a workaround for -0.0
//         req_grad = true;
//         this->context->save_for_backward({mask_tensor, t});
//     }
//     auto out = make_shared<tensor<float>>(*out_data, t->shape(), t->get_device(), req_grad);
//     return out;
// }

// template <>
// void cyg::Dropout<tensor<float>>::backward(std::valarray<float> *incoming_grad)
// {
//     assertm(this->isTrain, "pls set isTrain to true to backprop on this node");
//     if (!this->isTrain)
//         throw runtime_error("invalid op, pls enable isTrain to backprop");
//     auto var = this->context->get_variables();
//     assertm(var.size(), "invalid ops");
//     auto mask = var[0];
//     auto t = var[1];
//     if (t->require_grad())
//     {
//         // y = mask*t   dy/dt = mask
//         valarray<float> local_grad = 0.0f + *incoming_grad * *mask->data();
//         t->backward(&local_grad);
//     }
// }
