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
    using namespace std;

    template<class T>
    using tptr = shared_ptr<cyg::tensor<T>>;


    class Module
    {
    public:
        Module() {};
        bool training = true;
        string name;
        void register_module(string name, Module* module) { _modules.push_back({name, shared_ptr<Module>(module)}); }
        void register_parameter(string name, tptr<float> p) 
        {
            if(!p->requires_grad() || p->grad_fn) throw runtime_error("cannot add tensor as param, tensor requires_grad must be set to true and tensor must be non-leaf - explicitly created");
            _parameters[name] = p; 
        }
        void zero_grad()
        {
            for (auto [n, p] : _parameters)
            {
                p->zero_grad();
            }
            for (auto [n, m] : _modules)
            {
                m->zero_grad();
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
            for (auto [n, m] : _modules)
            {
                m->train(isTrain);
            }
        }
        tptr<float> get_parameter(string name)
        {
            if (_parameters.find(name) == _parameters.end())
                throw runtime_error("invalid input, no parameter with given name");
            
            return _parameters[name];
        }
        Module* get_module(string name) const
        {
            const auto it = find_if(_modules.begin(), _modules.end(), [&](const auto &p){ return get<0>(p)==name;});
            if(it==_modules.end()){
                throw runtime_error("invalid input, no mdule with given name");
            }
            return get<1>(*it).get();
        }
        tptr<float> operator()(const tptr<float> &input_tensor) { return forward(input_tensor); };
        virtual tptr<float> forward(const tptr<float> &input_tensor) { 
            throw runtime_error("not implemented");
        };
        virtual tuple<tptr<float>, tptr<float>, tptr<float>> forward( tuple<tptr<float>, tptr<float>, tptr<float>> inputs){
            throw runtime_error("not implemented");
        }
        Module(const Module &m) : _parameters(m._parameters), _modules(m._modules) {}; // rule of three/five/zero
        vector<pair<string, shared_ptr<Module>>> modules() const { return _modules; }
        vector<tptr<float>> parameters(const bool& recurse=true) const 
        {   
            vector<tptr<float>> res;
            if(_modules.size()==0) {
                for(const auto& [n, p]: _parameters) res.push_back(p);
            };
            for(const auto& [n, m]: _modules) {
                auto m_res = m->parameters(recurse);
                res.insert(res.end(), m_res.begin(), m_res.end());
            }
            return res;
        }
        ~Module()
        {
            _parameters.clear();
            _modules.clear();
        }

    protected:
        vector<pair<string, shared_ptr<Module>>> _modules;
        unordered_map<string, tptr<float>> _parameters;
    };

    ostream &operator<<(ostream &out, const vector<size_t> input)
    {
        out << "(";
        for (int i = 0; i < input.size() - 1; i++)
            out << input[i] << " , ";
        out << input[input.size() - 1];
        out << ")";
        return out;
    };

    // ostream &operator<<(ostream &out, const Module &module)
    // {
    //     auto modules = module.modules();
    //     if (!modules.empty())
    //     {
    //         for (const auto &[name, m] : modules)
    //         {
    //             out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
    //             out << "Module " << m->name << "\n";
    //             auto params = m->parameters();
    //             for (const auto &p : params)
    //             {
    //                 out << "Parameter " <<"\n"
    //                     << "size = ";
    //                 out << p->shape() << "\n";
    //             };
    //             out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
    //         };
    //     }
    //     auto params = module.parameters();
    //     if (!params.empty())
    //     {
    //         out << "Module " << module.name << "\n";
    //         out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
    //         out << "Parameters " << "\n";
    //         for (const auto &p : params)
    //         {
    //             out << p.first << "\t" << "size =( ";
    //             for (auto s : p.second->shape())
    //                 out << s << " ";
    //             out << ")\n";
    //         }
    //     }
    //     return out;
    // }

    class Linear : public Module
    {
        // y = input_tensor(dd * in) * w(out * in).transpose() + c(out * 1)
    public:
        Linear(size_t in_features, size_t out_features, bool bias = true) : Module(), _bias(bias), _in_features(in_features), _out_features(out_features)
        {
            this->name = "Linear";
            vector<size_t> weight_feat{out_features, in_features}, bias_feat{out_features};
            weight = make_shared<cyg::tensor<float>>(weight_feat, 1, true);
            register_parameter("weight", weight);
            if (_bias) {
                this->bias = make_shared<cyg::tensor<float>>(bias_feat, 1, true);
                register_parameter("bias",  this->bias);
            }
            reset_parameters();
        };
        // https://web.archive.org/web/20230129061039/http://github.com/torch/nn/blob/master/Linear.lua#L18
        // https://arxiv.org/pdf/1502.01852
        void reset_parameters()
        {
            const float bound = 1 / sqrt(_in_features);
            weight->uniform(-bound, bound);
            if (this->_bias)
                bias->uniform(-bound, bound);
        }
        virtual tptr<float> forward(const tptr<float> &input_tensor)
        {
            auto output = input_tensor->mm(weight->transpose(-1, -2));
            if (_bias) return output + _bias;
            return output;
        };

    tptr<float> weight, bias;
    bool _bias;
    size_t _in_features, _out_features;
    };

    class Sequential : public Module
    {
    public:
        Sequential() : Module() {};
        Sequential(vector<pair<string, Module*>> input) : Module()
        { // can also use vector of tuples
            for (const auto &[n, m] : input)
                register_module(n, m);
        };
        tptr<float> forward(const tptr<float> &input_tensor)
        {
            auto output = (*_modules[0].second)(input_tensor);
            for (auto m = next(_modules.begin()); m != _modules.end(); m++)
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
            if(p>1.0 || p<0.0) throw runtime_error("invalid input, prob should be between 0 and 1 (inclusive)");
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
            bernoulli_distribution d(1-p);
            auto mask_data = new valarray<bool>(input_tensor->numel());
            default_random_engine e(time(nullptr));

            generate(begin(*mask_data), end(*mask_data), [&]( ){ return d(e);});
            auto mask_tensor = make_shared<cyg::tensor<bool>>(input_tensor->shape(), mask_data);
            auto output = input_tensor->where(mask_tensor, 0.0f); // zero out neurons with prob of p (1-p in d)
            output = output / (1 - p); //scaled by 1/1-p
            output->grad_fn->name=name; //for debugging purpose

            return output;
        };
        float p;
    };
    shared_ptr<cyg::tensor<float>> softmax(const shared_ptr<cyg::tensor<float>> &input_tensor, int dim){

        auto sum_ = input_tensor->exp()->sum(dim, true);
        auto output = (input_tensor - sum_->log())->exp(); //numerical stability
        if(output->grad_fn) output->grad_fn->name="SoftmaxOp";

        return output;
    }
    class Softmax : public Module{
        public:
            Softmax(int d): Module(), _dim(d){ name="Softmax";}
             tptr<float> forward(const  tptr<float>& x){
                return softmax(x, _dim);
            };
        int _dim;
    };

//     // https://arxiv.org/abs/1502.03167
//     // https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
//     // normalizing layers to address covariate shift
    class BatchNorm : public Module{

        public:
            BatchNorm(size_t num_features, float eps=1e-05, float momentum=0.1, bool affine=true, bool track_running_stats=true): Module(), _num_features(num_features), _eps(eps), _momentum(momentum), _affine(affine), _tracking_running_stats(track_running_stats) {
                vector<size_t> dims = {1, num_features};
                auto _gammas = make_shared<cyg::tensor<float>>(dims, 1, true);
                register_parameter("gammas", _gammas);
                if (affine) {
                    auto _betas = make_shared<cyg::tensor<float>>(dims, 0, true);
                    register_parameter("betas", _betas);
                }
                if(_tracking_running_stats){
                    _mean_avg = make_shared<cyg::tensor<float>>(dims, 0, false); //stats, no backprop
                    _var_avg = make_shared<cyg::tensor<float>>(dims, 0, false);
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
        tptr<float> _gammas, _betas, _mean_avg, _var_avg;
        int _num_features;
        float _eps, _momentum;
        bool _affine, _tracking_running_stats;
    };


    class LayerNorm : public Module{
        public:
            LayerNorm(size_t normalized_shape, float eps=1e-05, bool elementwise_affine=true, bool bias=true): Module(), _normalized_shape(normalized_shape), _eps(eps), _elementwise_affine(elementwise_affine), _bias(bias){
                vector<size_t> dims = {1, normalized_shape};
                if (elementwise_affine) {
                    auto _gammas =  make_shared<cyg::tensor<float>>(dims, 1, true);
                    register_parameter("gammas", _gammas);
                    if(_bias){
                        auto _betas = make_shared<cyg::tensor<float>>(dims, 0, true);
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
             
        tptr<float> _gammas, _betas;
        int _normalized_shape;
        float _eps;
        bool _elementwise_affine, _bias;
    };

    shared_ptr<cyg::tensor<float>> tanh(const shared_ptr<cyg::tensor<float>>& x){
        auto shifted_x = x + 1e-12;
        auto n = shifted_x->exp() - (-shifted_x)->exp();
        auto d = shifted_x->exp() + (-shifted_x)->exp();
        auto output = n/d;
        if(output->grad_fn) output->grad_fn->name = "Tanh";
        return output;
    }

     shared_ptr<cyg::tensor<float>> sigmoid(const shared_ptr<cyg::tensor<float>> &x){
        auto shifted_x = x + 1e-12;
        auto output = (-shifted_x->exp() + 1)->pow(-1);
        if(output->grad_fn) output->grad_fn->name = "sigmoid";
        return output;
    };

    class Sigmoid : public Module{
        public:
            Sigmoid(): Module(){ name="Sigmoid";}
            tptr<float> forward(const tptr<float> &x){
                return sigmoid(x);
            };
    };

    class LogSoftmax : public Module{
        public:
            LogSoftmax(int dim): Module(), _dim(dim){ name="LogSoftmax";}
             tptr<float> forward(const  tptr<float>& x){
                auto output = softmax(x, _dim)->log();
                output->grad_fn->name=name;
                return output;
            }
        int _dim;
    };
    
    
    // https://arxiv.org/pdf/1609.04747
    class Optimizer
    {
        public:
            Optimizer(vector<tptr<float>> parameters): _parameters(parameters){}
            void zero_grad(){
                for(const auto& p:_parameters){
                    p->zero_grad();
                }
            };

        vector<tptr<float>> _parameters;
    };


    // sgd with momentum
    // see 4.1 of https://arxiv.org/pdf/1609.04747
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    class SGD : Optimizer
    {
        public:
            SGD(vector<tptr<float>> parameters, float lr, float momentum=0, float dampening=0, float weight_decay=0, bool nestorov=false): Optimizer(parameters), _lr(lr), _momentum(momentum), _dampening(dampening), _weight_decay(weight_decay), _nestorov(nestorov){}
            void step(){
                for(auto i = 0; i<_parameters.size();i++){
                    auto p = _parameters[i];
                    auto g_t = *p->grad();
                    if(_weight_decay!=0) g_t += (_weight_decay * *p->data());
                    if(_momentum!=0){
                        if(i>1) _velocity[i] = _momentum * _velocity[i] + (1-_dampening)*g_t;
                        else _velocity[i] = g_t; // torch initializes velocity to the gradients instead of zeros
                        if(_nestorov) g_t += _momentum*_velocity[i];
                        else g_t = _velocity[i];
                    }
                    valarray<float> ds = (*p->data()) - _velocity[i];
                    p->set_data(&ds);
                }
            }
        
        float _lr, _dampening, _momentum, _weight_decay;
        bool _nestorov;
        vector<valarray<float>> _velocity;
    };
    // see chapter 4.6
    class Adam : Optimizer
    {
        public:
            Adam(vector<tptr<float>> parameters, float lr, float b1, float b2, float eps=10-8): Optimizer(parameters), _lr(lr), _b1(b1), _b2(b2), _eps(eps){
                for(const auto& p : _parameters){
                    _velocity.push_back(valarray<float>(p->numel(), 0));
                    _momentum.push_back(valarray<float>(p->numel(), 0));
                }
            }
            void step(){
                for(auto i=0; i<_parameters.size();i++){
                    auto p = _parameters[i];
                    auto g = (*p->grad());
                    _momentum[i] = _b1*_momentum[i] + (1-_b1)*g;
                    _velocity[i] = _b2*_velocity[i] + (1-_b2)*pow(g, 2);
                    auto m_ = _momentum[i]/(1- pow(_b1, i+1));
                    auto v_ = _velocity[i]/(1- pow(_b2, i+1));
                    valarray<float> dg = g - (_lr * m_)/(sqrt(v_)*_eps);
                    p->set_data(&dg);
                }
            }
        float _lr, _b1, _b2, _eps;
        vector<valarray<float>> _velocity, _momentum;
    };
    // https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    // logits = N * C   targets = N
    shared_ptr<cyg::tensor<float>> cross_entropy_loss(const shared_ptr<cyg::tensor<float>> logits, const shared_ptr<cyg::tensor<int>> target){
        if(logits->rank()!=2 || target->rank()!=1) throw runtime_error("invalid input, logits must be of rank 2 and targets must be 1D tensor");
        auto x_n = logits->at(target); // N
        auto logits_exp = logits->exp();
        auto out = x_n->exp() / (logits_exp->sum(-1) + 1e-20);  // N / N => N
        out = -(out->log()); // N
        out = out->sum() / out->numel();
        out->grad_fn->name="CrossEntropy";
        return out;
    }
}
#endif