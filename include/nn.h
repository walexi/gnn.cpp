#ifndef NN_H
#define NN_H
#include "tensor.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <valarray>
#include <array>
#include <iostream>
#include <memory>
#include <numeric>
#include <assert.h>

namespace nn
{
    // grad=true, dtype=float

    class Module
    {
    protected:
        bool isTraining;
        std::unordered_map<std::string, std::shared_ptr<cyg::tensor<float>>> params; // params are tensor with req_grad=true
        std::unordered_map<std::string, std::shared_ptr<Module>> modules;            // @todo use template to handle both modules and params

    public:
        Module() {};
        virtual void add_params(std::vector<std::pair<std::string, std::shared_ptr<cyg::tensor<float>>>> input_params)
        {
            for (auto& p : input_params)
                this->params.insert(p); //@todo fix throw error
        };
        virtual void add_modules(std::vector<std::pair<std::string, std::shared_ptr<Module>>> input_modules)
        {
            for (auto& m : input_modules)
                modules.insert(m);
        };
        virtual void zero_grad()
        {
            for (auto &m : modules) m.second->zero_grad();
            for (auto &p : params) p.second->zero_grad();
        };
        virtual void eval() { train(false); };
        virtual void train(const bool &isTrain)
        {
            for (auto &p : params) p.second->enable_grad(isTrain);
            for (auto &m : modules) m.second->train(isTrain);//recursive
            isTraining = isTrain;
        };
        virtual std::shared_ptr<cyg::tensor<float>> operator()(const std::shared_ptr<cyg::tensor<float>> &input_tensor) { return this->forward(input_tensor); }; // you prolly dont wanna pass in a tensor without grad enabled
        virtual std::shared_ptr<cyg::tensor<float>> forward(const std::shared_ptr<cyg::tensor<float>> &input_tensor) { return std::shared_ptr<cyg::tensor<float>>(); };
        Module(const Module &m) : params(m.params), modules(m.modules) {}; // rule of three/five/zero
        virtual std::unordered_map<std::string, std::shared_ptr<cyg::tensor<float>>> parameters() const { return this->params; }
        virtual std::unordered_map<std::string, std::shared_ptr<Module>> children() const { return this->modules; }
        ~Module()
        {
            params.clear();
            modules.clear();
        }
    };

    class Linear : public Module
    {
        bool bias;
        size_t in_features;
        size_t out_features;
        // y = input_tensor(dd * in) * w(out * in).transpose() + c(out * 1)
    public:
        Linear(size_t in_features, size_t out_features, bool bias = true, cyg::Device device = cyg::Device::cpu) : Module(), bias(bias), in_features(in_features), out_features(out_features)
        {
            std::vector<size_t> weight_feat{out_features, in_features}, bias_feat{out_features};
            auto w = std::make_shared<cyg::tensor<float>>(weight_feat, 1, device, true);
            auto b =  std::make_shared<cyg::tensor<float>>(bias_feat, 1, device, true);
            this->add_params({{"weight", w}});
            if (bias) this->add_params({{"bias", b}});
            this->reset_parameters();
        };
        // https://web.archive.org/web/20230129061039/http://github.com/torch/nn/blob/master/Linear.lua#L18
        // https://arxiv.org/pdf/1502.01852
        // a silly/lazy implementation of kaiming initialization
        void reset_parameters()
        {
            const float bound = 1 / std::sqrt(in_features);
            this->params["weight"]->uniform(-bound, bound);
            if (this->bias) this->params["bias"]->uniform(-bound, bound);
        }
        std::shared_ptr<cyg::tensor<float>> forward(const std::shared_ptr<cyg::tensor<float>>& input_tensor)
        {
            auto output = input_tensor->mm(this->params["weight"]->transpose(-1, -2));
            if (this->bias)
            {
                for (int i = 0; i < output->rank() - 1; i++)
                {
                    this->params["bias"]->unsqueeze(0);
                    this->params["bias"]->repeat(0, output->shape()[i]);
                };
                return output + this->params["bias"];
            };
            return output;
        };
    };

    // class Sequential : Module
    // {
    // public:
    //     Sequential(std::vector<std::pair<std::string, std::shared_ptr<Module>>> input) : Module()
    //     { // can also use vector of tuples
    //         add_modules(input);
    //     };
    // };
    // };
    // class ReLU : public Module
    // {
    // public:
    //     ReLU() : Module() {};
    //     std::shared_ptr<cyg::tensor> forward(const std::shared_ptr<cyg::tensor> &input_tensor);
    // };

    // class Dropout : public Module
    // {
    //     int p;
    //     std::shared_ptr<tensor> drop;

    // public:
    //     Dropout(int p = 0.2) : Module(), p(p) {};
    //     std::shared_ptr<cyg::tensor> forward(const std::shared_ptr<cyg::tensor> &input_tensor) override;
    // }

    // public:
    //     Sequential(std::vector<std::shared_ptr<Module>> input_modules);
    //     template <class T>
    //     std::shared_ptr<cyg::tensor> operator()(const T &input_tensor);
    //     void add_module(string name, std::shared_ptr<Module> m);
    //     Sequential(const Sequential &seq) : modules(seq.modules) {} // rule of three/five/zero
    //     ~Sequential()
    //     {
    //         this->modules.reset();
    //     }
    // };

    // template <class T>
    // class ReLU : public Operation<T>
    // {
    // public:
    //     ReLU() : Operation<T>() {}
    //     std::shared_ptr<T> forward(const std::shared_ptr<T> &t);
    //     void backward(std::valarray<float> *incoming_grad) override;
    // };

    // template <class T>
    // class Dropout : public Operation<T>
    // {
    //     double p;
    //     bool isTrain;
    //     std::shared_ptr<T> drop;

    // public:
    //     Dropout(double p = 0.2, bool isTrain = true) : Operation<T>(), p(p), isTrain(isTrain) {}
    //     std::shared_ptr<T> forward(const std::shared_ptr<T> &t);
    //     void set_isTrain(bool isTrain) { this->isTrain = isTrain; };
    //     void backward(std::valarray<float> *incoming_grad) override;
    // };

    //     template<class T>
    //     class Softmax : public Operation<T>{
    //         int dim;
    //         std::shared_ptr<T> drop;
    //         public:
    //             Softmax(int d): Operation<T>(), dim(d){}
    //             std::shared_ptr<T> forward(const std::shared_ptr<T>& t);
    //             void backward(std::vector<float>* incoming_grad) override;
    //     };

    //     template<class T>
    //     class BatchNorm : public Operation<T>{

    //         public:
    //             BatchNorm(int num_features, float eps=1e-05, int momentum=0.1, bool affine=True, bool track_running_stats=True, cyg::Device device=Device::cpu): Operation<T>(), p(p){}
    //             std::shared_ptr<T> forward(const std::shared_ptr<T>& t);
    //             void backward(std::vector<float>* incoming_grad) override;
    //     };

    //     template<class T>
    //     class LayerNorm : public Operation<T>{
    //         std::shared_ptr<Parameter> weight;
    //         std::shared_ptr<Parameter> b;
    //         public:
    //             LayerNorm(std::vector<int>> normalized_shape, float eps=1e-05, bool elementwise_affine=True, bool bias=True,cyg::Device device=None,): Operation<T>(), p(p){}
    //             std::shared_ptr<T> forward(const std::shared_ptr<T>& t);
    //             void backward(std::vector<float>* incoming_grad) override;
    //     };

}
#endif