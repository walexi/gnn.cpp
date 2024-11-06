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
    //grad=true, dtype=float

    class Parameter : public cyg::tensor<float>
    {
    public:
        Parameter(std::vector<size_t> &dims, cyg::Device device = cyg::Device::cpu) : cyg::tensor<float>(dims, 1, device, true){};
    };

    class Module
    {
    protected:
        bool isTraining = true;
        std::unordered_map<std::string, std::shared_ptr<cyg::tensor<float>>> params; //params are tensor with req_grad=true

    public:
        Module() {};
        virtual void add_params(std::vector<std::pair<std::string, std::shared_ptr<cyg::tensor<float>>>> input_params){
            for (auto p : input_params) params.insert(p);
        };
        virtual void zero_grad(){  
            for (auto &p : params) p.second->zero_grad();
        };
        virtual void train(const bool& isTrain=true){
            for(auto& param : this->params){
                param.second->enable_grad(isTrain);
            }
        };
        virtual void eval(){ this->train(false);};
        virtual std::shared_ptr<cyg::tensor<float>> operator()(const std::shared_ptr<cyg::tensor<float>> &input_tensor) { return this->forward(input_tensor); };//you prolly dont wanna pass in a tensor without grad enabled
        virtual std::shared_ptr<cyg::tensor<float>> forward(const std::shared_ptr<cyg::tensor<float>>& input_tensor){ return std::shared_ptr<cyg::tensor<float>>();};
        Module(const Module &m) : params(m.params) {}; // rule of three/five/zero
        virtual std::unordered_map<std::string, std::shared_ptr<cyg::tensor<float>>> parameters() const { return this->params; }
        ~Module(){params.clear();}
    };

    class Linear : public Module
    {
        bool bias;
        // y = input_tensor(dd * in) * w(out * in).transpose() + c(out * 1)
    public:
        Linear(size_t in_features, size_t out_features, bool bias = true, cyg::Device device = cyg::Device::cpu): Module(), bias(bias){
            std::vector<size_t> weight_feat {out_features, in_features}, bias_feat {out_features};
            this->add_params({make_pair("weight", std::make_shared<cyg::tensor<float>>(weight_feat, 1, device, true))});
            if(bias) this->add_params({make_pair("bias", std::make_shared<cyg::tensor<float>>(bias_feat, 1, device, true))});
            this->reset_parameters();
        };
        // https://web.archive.org/web/20230129061039/http://github.com/torch/nn/blob/master/Linear.lua#L18
        // https://arxiv.org/pdf/1502.01852
        // a silly/lazy implementation of kaiming initialization
        void reset_parameters(){
            const float bound = 1/std::sqrt(this->params["weights"]->shape()[this->params["weights"]->rank()-1]);
            this->params["weights"]->uniform(-bound, bound);
            if(this->bias) this->params["bias"]->uniform(-bound, bound);
        }
        std::shared_ptr<cyg::tensor<float>> forward(const std::shared_ptr<cyg::tensor<float>> &input_tensor){
            auto w = this->params["weights"];
            auto output = input_tensor->mm(w->transpose(w->rank()-1, w->rank()-2));
            if(this->bias) {
                for(int i=0;i<output->rank()-1; i++) {
                    this->params["bias"]->unsqueeze(0);
                    this->params["bias"]->repeat(0, output->shape()[i]);
                };
                return output + this->params["bias"];
            };
            return output;
        };
    };



}
#endif