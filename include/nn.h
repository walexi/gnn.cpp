#ifndef NN_H
#define NN_H
#include "tensor.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <valarray>
#include <iostream>
#include <memory>
#include <numeric>
#include <assert.h>
#include <random>

namespace nn
{
    // grad=true, dtype=float

    // class Parameter : public cyg::tensor<float>
    // {
    //     public:
    //         Parameter(std::vector<size_t> dims, int value=0) : cyg::tensor<float>(dims, value, true){}
    //         cyg::tensor<float> *operator->() {
    //             return this;
    //         }
    // };

    class Module
    {
    public:
        Module() {};
        bool training = true;
        std::string name;
        void register_module(std::string name, Module *module) { _modules.push_back({name, std::shared_ptr<Module>(module)}); }
        void register_parameter(std::string name, cyg::tptr<float> p);
        void register_buffer(std::string name, cyg::tptr<float> p) { _buffers[name] = p; };
        void zero_grad();
        void eval() { train(false); };
        void train(const bool &isTrain = true);
        cyg::tptr<float> get_parameter(std::string name);
        cyg::tptr<float> get_buffer(std::string name);
        Module *get_module(std::string name);
        cyg::tptr<float> operator()(const cyg::tptr<float> &input_tensor, cyg::tensor<int> *y = nullptr);
        virtual cyg::tptr<float> forward(const cyg::tptr<float> &input_tensor) { throw std::runtime_error("not implemented"); };
        virtual cyg::tptr<float> forward(const cyg::tptr<float> &input_tensor, cyg::tensor<int> *y) { throw std::runtime_error("not implemented"); };
        Module(const Module &m) : _modules(m._modules), _parameters(m._parameters), _buffers(m._buffers) {}; // rule of three/five/zero
        std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules(const bool &recurse = true) const;
        std::vector<cyg::tptr<float>> parameters(const bool &recurse = true) const;
        std::vector<cyg::tptr<float>> buffers(const bool &recurse = true) const; // can be float, int, bool, trying to avoid templating Module, btw any other type can be easily cast to float
        ~Module();
    
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> _modules;
    std::unordered_map<std::string, cyg::tptr<float>> _parameters;
    std::unordered_map<std::string, cyg::tptr<float>> _buffers;
};

    class Linear : public Module
    {
        // y = input_tensor(dd * in) * w(out * in).t() + c(out * 1)
        public:
            Linear(size_t in_features, size_t out_features, bool bias = true);
            void reset_parameters();
            cyg::tptr<float> forward(const cyg::tptr<float> &input_tensor) override;

        bool _bias;
        size_t _in_features, _out_features;
    };

    class Sequential : public Module
    {
    public:
        Sequential() : Module() {};
        Sequential(std::vector<std::pair<std::string, Module *>> input);
        void add_module(std::string n, Module* m){ register_module(n, m);}
        cyg::tptr<float> forward(const cyg::tptr<float> &input_tensor) override;
    };

    // https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    // prevents non-saturation of gradients, speeds up sgd
    class ReLU : public Module
    {
        public:
            ReLU() : Module() { name = "ReLU"; }
            cyg::tptr<float> forward(const cyg::tptr<float> &input_tensor) override;
    };

    // https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    class Dropout : public Module
    {
        public:
            Dropout(float p = 0.2);
            cyg::tptr<float> forward(const cyg::tptr<float> &input_tensor) override;
        float p;
    };

    cyg::tptr<float> softmax(const cyg::tptr<float> &input_tensor, int dim);
    
    class Softmax : public Module
    {
        public:
            Softmax(int d) : Module(), _dim(d) { name = "Softmax"; }
            cyg::tptr<float> forward(const cyg::tptr<float> &x) override;
        int _dim;
    };

    //     // https://arxiv.org/abs/1502.03167
    //     // normalizing layers to address covariate shift
    class BatchNorm : public Module
    {
        public:
            BatchNorm(size_t num_features, float eps = 1e-05, float momentum = 0.1, bool affine = true, bool track_running_stats = true);
            cyg::tptr<float> forward(const cyg::tptr<float> &x) override;

        int _num_features;
        float _eps, _momentum;
        bool _affine, _tracking_running_stats;
    };

    class LayerNorm : public Module
    {
        public:
            LayerNorm(size_t normalized_shape, float eps = 1e-05, bool elementwise_affine = true, bool bias = true);
            cyg::tptr<float> forward(const cyg::tptr<float> &x) override;

        int _normalized_shape;
        float _eps;
        bool _elementwise_affine, _bias;
    };

    cyg::tptr<float> tanh(const cyg::tptr<float> &x);
    cyg::tptr<float> sigmoid(const cyg::tptr<float> &x);

    class Sigmoid : public Module
    {
        public:
            Sigmoid() : Module() { name = "Sigmoid"; }
            cyg::tptr<float> forward(const cyg::tptr<float> &x) override;
    };

    class LogSoftmax : public Module
    {
        public:
            LogSoftmax(int dim) : Module(), _dim(dim) { name = "LogSoftmax"; }
            cyg::tptr<float> forward(const cyg::tptr<float> &x) override;

        int _dim;
    };

    // https://arxiv.org/pdf/1609.04747
    class Optimizer
    {
        public:
            Optimizer(std::vector<cyg::tptr<float>> parameters) : _parameters(parameters) {}
            void zero_grad();

        std::vector<cyg::tptr<float>> _parameters;
    };

    // sgd with momentum
    // see 4.1 of https://arxiv.org/pdf/1609.04747
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    class SGD : Optimizer
    {
        public:
            SGD(std::vector<cyg::tptr<float>> parameters, float lr, float momentum = 0, float dampening = 0, float weight_decay = 0, bool nestorov = false)
                : Optimizer(parameters), _lr(lr), _momentum(momentum), _dampening(dampening), _weight_decay(weight_decay), _nestorov(nestorov) {}
            void step();

        float _lr, _dampening, _momentum, _weight_decay;
        bool _nestorov;
        std::vector<std::valarray<float>> _velocity;
    };
    // see chapter 4.6
    class Adam : Optimizer
    {
        public:
            Adam(std::vector<cyg::tptr<float>> parameters, float lr, float b1, float b2, float eps = 10 - 8);
            void step();

        float _lr, _b1, _b2, _eps;
        std::vector<std::valarray<float>> _velocity, _momentum;
    };
    // https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    // logits = N * C   targets = N
    cyg::tptr<float> cross_entropy_loss(const cyg::tptr<float> logits, const cyg::tptr<int> target);

    class MLP: public Module
    {
        public:
            MLP(size_t in_channel, std::vector<size_t> hid_dims, const bool& bias=true, const float& dropout=0.0){
                auto seq = new Sequential();
                for(auto i=0; auto hid_dim:hid_dims){
                    auto i_s = std::to_string(i);
                    seq->add_module("lin_"+ i_s, new Linear(in_channel, hid_dim, bias));
                    if(hid_dim!=hid_dims[hid_dims.size()-1])
                    { 
                        seq->add_module("lnorm_"+i_s, new LayerNorm(hid_dim));
                        seq->add_module("relu_"+i_s, new ReLU());
                    }
                    seq->add_module("drop_"+ i_s, new Dropout(dropout));
                    in_channel = hid_dim; i++;
                }
                register_module("seq", seq);
            }
            cyg::tptr<float> forward(const cyg::tptr<float> &input){
                return (*get_module("seq"))(input);
            };
    };
}
#endif