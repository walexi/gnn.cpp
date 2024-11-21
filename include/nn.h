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
    template<class T>
    using tptr = std::shared_ptr<cyg::tensor<T>>;

    class Module : public std::enable_shared_from_this<Module>
    {
    public:
        Module() {};
        bool training = true;
        std::string name;
        void register_module(std::string name, Module* module) { _modules.push_back({name, std::shared_ptr<Module>(module)}); }
        void register_parameter(std::string name, tptr<float> p) { _parameters[name] = p; }
        void zero_grad()
        {
            for (auto [n, p] : _parameters)
            {
                p->zero_grad();
            }
            for (auto m : _modules)
            {
                m.second->zero_grad();
            }
        };
        void eval() { train(false); };
        void train(const bool &isTrain = true)
        {
            training = isTrain;
            for (auto [n, p] : _parameters)
            {
                p->requires_grad_(isTrain);
            }
            for (auto m : _modules)
            {
                m.second->train(isTrain);
            }
        }
        tptr<float> get_parameter(std::string name)
        {
            if (_parameters.find(name) == _parameters.end())
                throw std::runtime_error("invalid input, no parameter with given name");
            return _parameters[name];
        }
        tptr<float> operator()(const tptr<float> &input_tensor) { return forward(input_tensor); }; // you prolly dont wanna pass in a tensor without grad enabled
        virtual tptr<float> forward(const  tptr<float> &input_tensor) { return tptr<float>(); };
        Module(const Module &m) : _parameters(m._parameters), _modules(m._modules) {}; // rule of three/five/zero
        std::vector<std::pair<std::string, std::shared_ptr<Module>>> modules() const { return _modules; }
        std::unordered_map<std::string, tptr<float>> parameters() const { return _parameters; }
        ~Module()
        {
            _parameters.clear();
            _modules.clear();
        }

    protected:
        std::vector<std::pair<std::string, std::shared_ptr<Module>>> _modules; // tuple/pair
        std::unordered_map<std::string, tptr<float>> _parameters;
    };

    std::ostream &operator<<(std::ostream &out, const std::vector<size_t> input)
    {
        out << "(";
        for (int i = 0; i < input.size() - 1; i++)
            out << input[i] << " , ";
        out << input[input.size() - 1];
        out << ")";
        return out;
    };

    std::ostream &operator<<(std::ostream &out, const Module &module)
    {
        auto modules = module.modules();
        if (!modules.empty())
        {
            for (const auto &[name, m] : modules)
            {
                out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
                out << "Module " << m->name << "\n";
                auto params = m->parameters();
                for (const auto &p : params)
                {
                    out << "Parameter " << p.first << "\n"
                        << "size = ";
                    out << p.second->shape() << "\n";
                };
                out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
            };
        }
        auto params = module.parameters();
        if (!params.empty())
        {
            out << "Module " << module.name << "\n";
            out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
            out << "Parameters " << "\n";
            for (const auto &p : params)
            {
                out << p.first << "\t" << "size =( ";
                for (auto s : p.second->shape())
                    out << s << " ";
                out << ")\n";
            }
        }
        return out;
    }

    class Linear : public Module
    {
        // y = input_tensor(dd * in) * w(out * in).transpose() + c(out * 1)
    public:
        tptr<float> weight;
        tptr<float> bias = nullptr;

        Linear(size_t in_features, size_t out_features, bool bias = true) : Module(), _bias(bias), _in_features(in_features), _out_features(out_features)
        {
            this->name = "Linear";
            std::vector<size_t> weight_feat{out_features, in_features}, bias_feat{out_features};
            weight = make_shared<cyg::tensor<float>>(weight_feat, 1, true);
            register_parameter("weight", weight);
            if (bias)
            {
                this->bias = make_shared<cyg::tensor<float>>(bias_feat, 1, true);
                register_parameter("bias", this->bias);
            }
            reset_parameters();
        };
        // https://web.archive.org/web/20230129061039/http://github.com/torch/nn/blob/master/Linear.lua#L18
        // https://arxiv.org/pdf/1502.01852
        void reset_parameters()
        {
            const float bound = 1 / std::sqrt(_in_features);
            weight->uniform(-bound, bound);
            if (this->_bias)
                bias->uniform(-bound, bound);
        }
         tptr<float> forward(const  tptr<float> &input_tensor)
        {
            auto output = input_tensor->mm(weight->transpose(-1, -2));
            if (this->_bias)
                return output + bias;
            return output;
        };

        bool _bias;
        size_t _in_features;
        size_t _out_features;
    };

    class Sequential : public Module
    {
    public:
        Sequential() : Module() {};
        Sequential(std::vector<std::pair<std::string, Module *>> input) : Module()
        { // can also use vector of tuples
            for (const auto &[n, m] : input)
                register_module(n, m);
        };
         tptr<float> forward(const  tptr<float> &input_tensor)
        {
            auto output = (*_modules[0].second)(input_tensor);
            for (auto m = std::next(_modules.begin()); m != _modules.end(); m++)
            {
                output = (*m->second)(output);
            }
            return output;
        };
    };
    // https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    // prevents non-saturation of gradients, speeds up sgd
    class ReLU : public Module
    {
    public:
        ReLU() : Module() { name = "ReLU"; }
         tptr<float> forward(const  tptr<float> &input_tensor)
        {
            auto condition = input_tensor > 0.0f;
            // auto mask = zeros->where(input_tensor>0.0f, 1.0f); // max(0, x)  y = x if x>0 else 0;
            // auto output = functional::abs(input_tensor * mask); //using abs to handle -ve float, no side effect on backprop of input_tensor
            auto output = input_tensor->where(condition, 0.0f);
            output->grad_fn->name=name;
            return output;
        };
    };

    // https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
    class Dropout : public Module
    {
    public:
        Dropout(float p = 0.2) : Module(), p(p) {
            if(p>1.0 || p<0.0) throw std::runtime_error("invalid input, prob should be between 0 and 1 (inclusive)");
            training = true;
            name="DropoutOp";
        };
         tptr<float> forward(const  tptr<float> &input_tensor)
        {
            if(!training){
                return input_tensor;
            }
            /**
             * https://en.cppreference.com/w/cpp/numeric/random/bernoulli_distribution
             * give "true"(1) 1-p of the time 
             * give "false"(0) p of the time
             * */
            std::bernoulli_distribution d(1-p);
            auto mask_data = new std::valarray<bool>(input_tensor->numel());
            std::generate(std::begin(*mask_data), std::end(*mask_data), [&]( ){ return d(e);});
            auto mask_tensor = make_shared<cyg::tensor<bool>>(input_tensor->shape(), mask_data);
            auto output = input_tensor->where(mask_tensor, 0.0f); // zero out neurons with prob of p (1-p in d)
            output = output / (1 - p); //scaled by 1/1-p
            output->grad_fn->name=name;

            return output;
        };
        float p;
    };

    class Softmax : public Module{
        public:
            Softmax(int d): Module(), _dim(d){ name="Softmax";}
             tptr<float> forward(const  tptr<float>& x){
                auto output = functional::softmax(x, _dim);
                return output;
            };
        int _dim;
    };

    // https://arxiv.org/abs/1502.03167
    // https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
    // normalizing layers to address covariate shift
    class BatchNorm : public Module{

        public:
            BatchNorm(size_t num_features, float eps=1e-05, float momentum=0.1, bool affine=true, bool track_running_stats=true): Module(), _num_features(num_features), _eps(eps), _momentum(momentum), _affine(affine), _tracking_running_stats(track_running_stats) {
                std::vector<size_t> dims = {1, num_features};
                _gammas = std::make_shared<cyg::tensor<float>>(dims, 1, true);
                register_parameter("gammas", _gammas);
                if (affine) {
                    _betas = std::make_shared<cyg::tensor<float>>(dims, 0, true);
                    register_parameter("betas", _betas);
                }
                if(_tracking_running_stats){
                    _mean_avg = std::make_shared<cyg::tensor<float>>(dims, 0, false); //stats, no backprop
                    _var_avg = std::make_shared<cyg::tensor<float>>(dims, 0, false);
                }
                training = true;
                name="BatchNormOp";
            }
             tptr<float> forward(const  tptr<float>& x){
                
                auto mean = x->mean(-2, true); // mean of features across batch _,N,F
                 tptr<float> scaled_x;
                if(!training && _tracking_running_stats){
                    // assertm(!x->_enable_grad, "pls disbale grad to run inference");
                    scaled_x = (x - _mean_avg) / (_var_avg + _eps)->pow(0.5);
                } else {
                    auto var = x->var(-2, 0, true);
                    scaled_x = (x- mean) / (var + _eps)->pow(0.5);
                }
                auto scaled_output = scaled_x * _gammas;
                if(_affine) scaled_output = scaled_output + _betas;

                if(_tracking_running_stats && training) {
                    mean->requires_grad_(false);  //temporaily disable autograd to compute stats below, not neccessary though
                    x->requires_grad_(false);
                    _mean_avg = (_mean_avg * _momentum) + mean * (1 - _momentum);
                    _var_avg = (_var_avg * _momentum) + x->var(-2, 1, true, false) * (1 - _momentum);
                    x->requires_grad_(true);
                    mean->requires_grad_(true); //prolly not neccessarily, just some bookkeeping
                }
                scaled_output->grad_fn->name = name;
                return scaled_output;
            }
        // prolly not needed, can as well use the get_parameter method
         tptr<float> _gammas;
         tptr<float> _betas;
         tptr<float> _mean_avg;
         tptr<float> _var_avg;
        int _num_features;
        float _eps;
        float _momentum;
        bool _affine;
        bool _tracking_running_stats;
    };


    class LayerNorm : public Module{
        public:
            LayerNorm(size_t normalized_shape, float eps=1e-05, bool elementwise_affine=true, bool bias=true): Module(), _normalized_shape(normalized_shape), _eps(eps), _elementwise_affine(elementwise_affine), _bias(bias){
                std::vector<size_t> dims = {1, normalized_shape};
                if (elementwise_affine) {
                    _gammas = std::make_shared<cyg::tensor<float>>(dims, 1, true);
                    register_parameter("gammas", _gammas);
                    if(_bias){
                        _betas = std::make_shared<cyg::tensor<float>>(dims, 0, true);
                        register_parameter("betas", _betas);
                    }
                }
                training = true;
                name="LayerNormOp";
            }
             tptr<float> forward(const  tptr<float>& x){
                auto scaled_x = (x - x->mean(-1, true)) / (x->var(-1, 0, true) + _eps)->pow(0.5);
                if(_elementwise_affine) scaled_x = scaled_x * _gammas;
                if(_bias) scaled_x = scaled_x + _betas;
                scaled_x->grad_fn->name=name;
                return scaled_x;
            }
             
         tptr<float> _gammas;
         tptr<float> _betas;
        int _normalized_shape;
        float _eps;
        bool _elementwise_affine;
        bool _bias;
    };

    class LogSoftmax : public Module{
        public:
            LogSoftmax(int dim): Module(), _dim(dim){ name="LogSoftmax";}
             tptr<float> forward(const  tptr<float>& x){
                auto output = functional::log(functional::softmax(x, _dim));
                output->grad_fn->name=name;
                return output;
            }
        int _dim;
    };
    

}
#endif