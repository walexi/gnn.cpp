#include <memory>
#include <vector>
#include <ranges>
#include "nn.h"
#include "operation.h"
#include "util.h"

using namespace nn;
using namespace std;
using namespace cyg;

void nn::Module::register_parameter(string name, tptr<float> p)
{
    if (!p->requires_grad() || p->grad_fn)
        throw runtime_error("cannot add tensor as param, tensor requires_grad must be set to true and tensor must be non-leaf - explicitly created");
    _parameters[name] = p;
}
void nn::Module::register_buffer(std::string name, cyg::tptr<float> b) {
    if (b->requires_grad() || b->grad_fn)
        throw runtime_error("cannot add tensor as buffer, tensor requires_grad must be set to false and tensor must be non-leaf - explicitly created");
    _buffers[name] = b;
};

void nn::Module::zero_grad()
{
    for (auto [n, p] : _parameters)
    {
        p->zero_grad();
    }
    for (auto [n, m] : _modules)
    {
        m->zero_grad();
    }
}
void nn::Module::train(const bool &isTrain)
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
cyg::tptr<float> nn::Module::get_parameter(std::string name)
{
    auto params = named_parameters();
    if (params.find(name) == params.end())
        throw runtime_error("invalid input, no parameter with given name");

    return params[name];
}
cyg::tptr<float> nn::Module::get_buffer(std::string name)
{
    auto buffers = named_buffers();
    if (buffers.find(name) == buffers.end())
        throw runtime_error("invalid input, no buffer with given name");

    return buffers[name];
}
std::shared_ptr<Module> nn::Module::get_module(std::string name)
{
    auto modules = named_modules();
    // std::cout<<"size="<<modules.size()<<"\n";
    if (modules.find(name) == modules.end())
        throw runtime_error("invalid input, no Module with given name");
    return modules[name];
}
cyg::tptr<float> nn::Module::operator()(const cyg::tptr<float> &input_tensor, cyg::tensor<int> *y)
{
    if (y == nullptr)
        return forward(input_tensor);
    return forward(input_tensor, y);
};

std::vector<std::shared_ptr<Module>> nn::Module::modules(const bool &recurse)
{
    auto n_modules = named_modules(recurse);
    vector<std::shared_ptr<Module>> modules;
    for(const auto &[n, m]: n_modules) modules.push_back(m);
    
    return modules;
}

std::unordered_map<std::string, std::shared_ptr<Module>> nn::Module::named_modules(const bool &recurse)
{
    std::unordered_map<std::string, std::shared_ptr<Module>> res;
    if(!recurse || _modules.size()==0) return {{name, this->shared_from_this()}};

    for (const auto &[n, m] : _modules)
    {
        auto m_res = m->named_modules(recurse);
        for(const auto& [n_m, m_]: m_res){
            if(res.contains(n_m))
                res[n + "_" + n_m] = m_;
            else res[n_m]=m_;
        }
    };
    return res;
}
vector<tptr<float>> nn::Module::parameters(const bool &recurse)
{
    auto n_params = named_parameters(recurse);
    vector<tptr<float>> params;
    for(const auto &[n,p]: n_params) params.push_back(p);
    return params;
};
std::unordered_map<std::string, cyg::tptr<float>> nn::Module::named_parameters(const bool &recurse)
{
    std::unordered_map<std::string, cyg::tptr<float>> res = _parameters;
    if(!recurse || _modules.size()==0) return res;

    for (const auto &[n, m] : _modules)
    {
        auto m_res = m->named_parameters(recurse);
        for(const auto& [n_p, p]: m_res) {
            if(res.contains(n_p))
                res[n + "_" + n_p] = p;
            else res[n_p]=p;
        }
    };
    return res;
};

std::unordered_map<std::string, cyg::tptr<float>> nn::Module::named_buffers(const bool &recurse) const
{
    std::unordered_map<std::string, cyg::tptr<float>> res = _buffers;
    if(!recurse || _modules.size()==0) return res;
    
    for (const auto &[n, m] : _modules)
    {
        auto m_res = m->named_buffers(recurse);
        for(const auto& [n_p, b]: m_res) res[n + "_" + n_p] = b;
    };
    return res;
};
std::vector<cyg::tptr<float>> nn::Module::buffers(const bool &recurse) const
{
    auto n_buffers = named_buffers(recurse);
    vector<tptr<float>> buffers;
    for(const auto &[n, b]: n_buffers) buffers.push_back(b);

    return buffers;
}
nn::Module::~Module()
{
    _parameters.clear();
    _modules.clear();
};

ostream &operator<<(ostream &out, Module &module)
{
    auto modules = module.named_modules();
    if (!modules.empty())
    {
        for (const auto &[name, m] : modules)
        {
            out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
            out << "Module " << m->name << "\n";
            auto params = m->parameters();
            for (const auto &p : params)
            {
                out << "Parameter " << "\n"
                    << "size = ";
                out << p->shape() << "\n";
            };
            out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
        };
    }
    auto params = module.named_parameters();
    if (!params.empty())
    {
        out << "Module " << module.name << "\n";
        out << "<<<<<<<<<<<<<<<<<<<<<<<<<" << "\n";
        out << "Parameters " << "\n";
        for (const auto &[n, p] : params)
        {
            out << n<<" ";
            out << p->shape() << "\n";
        }
    }
    return out;
};

nn::Linear::Linear(const size_t &in_features, const size_t &out_features, const bool &bias, const std::string &n) : Module(n), _bias(bias), _in_features(in_features), _out_features(out_features)
{
    vector<size_t> weight_feat{out_features, in_features}, bias_feat{out_features};
    register_parameter("weight", make_shared<cyg::tensor<float>>(weight_feat, 1, true));
    if (_bias)
        register_parameter("bias", make_shared<cyg::tensor<float>>(bias_feat, 1, true));
    reset_parameters();
};

// https://web.archive.org/web/20230129061039/http://github.com/torch/nn/blob/master/Linear.lua#L18
// https://arxiv.org/pdf/1502.01852
void nn::Linear::reset_parameters()
{
    const float bound = 1 / sqrt(_in_features);
    get_parameter("weight")->uniform(-bound, bound);
    if (this->_bias)
        get_parameter("bias")->uniform(-bound, bound);
}
tptr<float> nn::Linear::forward(const tptr<float> &input_tensor)
{
    auto output = input_tensor->mm(get_parameter("weight")->t(-1, -2));
    if (_bias)
        return output + get_parameter("bias");
    return output;
};

nn::Sequential::Sequential(std::vector<std::pair<std::string, Module *>> input, const std::string &n): Module(n)
{ // can also use std::vector of tuples
    for (const auto &[n, m] : input)
        register_module(n, m);
};

tptr<float> nn::Sequential::forward(const tptr<float> &input_tensor)
{
    auto output = (*_modules[0].second)(input_tensor);
    for (auto m = next(_modules.begin()); m != _modules.end(); m++)
    {
        output = (*m->second)(output);
    }
    return output;
};

cyg::tptr<float> nn::ReLU::forward(const cyg::tptr<float> &input_tensor)
{
    auto condition = input_tensor > 0.0;
    std::cout<<condition->shape()<<"\n";
    // auto mask = zeros->where(input_tensor>0.0f, 1.0f); // max(0, x)  y = x if x>0 else 0;
    // auto output = functional::abs(input_tensor * mask); //using abs to handle -ve float, no side effect on backprop of input_tensor
    auto output = input_tensor->where(condition, 0.0f);
    output->grad_fn->name = name;
    return output;
};

nn::Dropout::Dropout(const float &p, const std::string &n) : Module(n), p(p)
{
    if (p > 1.0 || p < 0.0)
        throw runtime_error("invalid input, prob should be between 0 and 1 (inclusive)");
    training = true;
}
cyg::tptr<float> nn::Dropout::forward(const cyg::tptr<float> &input_tensor)
{
    if (!training)
    {
        return input_tensor;
    }
    /**
     * https://en.cppreference.com/w/cpp/numeric/random/bernoulli_distribution
     * give "true"(1) 1-p of the time
     * give "false"(0) p of the time
     * */
    bernoulli_distribution d(1 - p);
    auto mask_data = new valarray<bool>(input_tensor->numel());
    default_random_engine e(time(nullptr));

    generate(begin(*mask_data), end(*mask_data), [&]()
             { return d(e); });
    auto mask_tensor = make_shared<cyg::tensor<bool>>(input_tensor->shape(), mask_data);
    auto output = input_tensor->where(mask_tensor, 0.0f); // zero out neurons with prob of p (1-p in d)
    output = output / (1 - p);                            // scaled by 1/1-p
    output->grad_fn->name = name;                         // for debugging purpose

    return output;
};

cyg::tptr<float> nn::softmax(const cyg::tptr<float> &input_tensor, int dim)
{
    auto sum_ = input_tensor->exp()->sum(dim, true);
    auto output = (input_tensor - sum_->log())->exp(); // numerical stability
    if (output->grad_fn)
        output->grad_fn->name = "SoftmaxOp";

    return output;
}

cyg::tptr<float> nn::Softmax::forward(const cyg::tptr<float> &x)
{
    return softmax(x, _dim);
};

nn::BatchNorm::BatchNorm(const size_t &num_features, const float &eps, const float &momentum, const bool &affine, const bool &track_running_stats, const std::string &n) : Module(n), _num_features(num_features), _eps(eps), _momentum(momentum), _affine(affine), _tracking_running_stats(track_running_stats)
{
    std::vector<size_t> dims = {1, num_features};
    register_parameter("gammas", make_shared<cyg::tensor<float>>(dims, 1, true));
    if (affine)
    {
        register_parameter("betas", make_shared<cyg::tensor<float>>(dims, 0, true));
    }
    if (_tracking_running_stats)
    {
        register_buffer("running_mean", make_shared<cyg::tensor<float>>(dims, 0, false)); // stats, no backprop
        register_buffer("running_var", make_shared<cyg::tensor<float>>(dims, 0, false));
    }
    training = true;
};

cyg::tptr<float> nn::BatchNorm::forward(const cyg::tptr<float> &x)
{
    auto mean = x->mean(-2, true); // mean of features across batch _,N,F
    cyg::tptr<float> scaled_x;
    if (!training && _tracking_running_stats)
    {
        // assertm(!x->_enable_grad, "pls disbale grad to run inference");
        scaled_x = (x - get_buffer("running_mean")) / (get_buffer("running_var") + _eps)->pow(0.5);
    }
    else
    {
        auto var = x->var(-2, 0, true);
        scaled_x = (x - mean) / (var + _eps)->pow(0.5);
    }
    auto scaled_output = scaled_x * get_parameter("gammas");
    if (_affine)
        scaled_output = scaled_output + get_parameter("betas");

    if (_tracking_running_stats && training)
    {
        mean->requires_grad_(false); // temporaily disable autograd to compute stats below, not neccessary though
        x->requires_grad_(false);
        get_buffer("running_mean") = (get_buffer("running_mean") * _momentum) + mean * (1 - _momentum);
        get_buffer("running_var") = (get_buffer("running_var") * _momentum) + x->var(-2, 1, true, false) * (1 - _momentum);
        x->requires_grad_(true);
        mean->requires_grad_(true); // prolly not neccessarily, just some bookkeeping
    }
    scaled_output->grad_fn->name = name;
    return scaled_output;
};

nn::LayerNorm::LayerNorm(const size_t &normalized_shape, const float &eps,const bool &elementwise_affine, const bool &bias, const std::string &n) : Module(n), _normalized_shape(normalized_shape), _eps(eps), _elementwise_affine(elementwise_affine), _bias(bias)
{
    std::vector<size_t> dims = {1, normalized_shape};
    if (elementwise_affine)
    {
        register_parameter("gammas", make_shared<cyg::tensor<float>>(dims, 1, true));
        if (_bias)
            register_parameter("betas", make_shared<cyg::tensor<float>>(dims, 0, true));
    }
    training = true;
};

cyg::tptr<float> nn::LayerNorm::forward(const cyg::tptr<float> &x)
{
    auto scaled_x = (x - x->mean(-1, true)) / (x->var(-1, 0, true) + _eps)->pow(0.5);
    if (_elementwise_affine)
        scaled_x = scaled_x * get_parameter("gammas");
    if (_bias)
        scaled_x = scaled_x + get_parameter("betas");
    scaled_x->grad_fn->name = name;
    return scaled_x;
};

inline cyg::tptr<float> nn::tanh(const cyg::tptr<float> &x)
{
    auto shifted_x = x + 1e-12;
    auto n = shifted_x->exp() - (-shifted_x)->exp();
    auto d = shifted_x->exp() + (-shifted_x)->exp();
    auto output = n / d;
    if (output->grad_fn)
        output->grad_fn->name = "Tanh";
    return output;
};

inline cyg::tptr<float> nn::sigmoid(const cyg::tptr<float> &x)
{
    auto shifted_x = x + 1e-12;
    auto output = (-shifted_x->exp() + 1)->pow(-1);
    if (output->grad_fn)
        output->grad_fn->name = "sigmoid";
    return output;
}

cyg::tptr<float> nn::Sigmoid::forward(const cyg::tptr<float> &x)
{
    return sigmoid(x);
};

cyg::tptr<float> nn::LogSoftmax::forward(const cyg::tptr<float> &x)
{
    auto output = softmax(x, _dim)->log();
    output->grad_fn->name = name;
    return output;
};

void nn::Optimizer::zero_grad()
{
    for (const auto &p : _parameters)
    {
        p->zero_grad();
    }
};

void nn::SGD::step()
{
    for (auto i = 0; i < _parameters.size(); i++)
    {
        auto p = _parameters[i];
        auto g_t = *p->grad();
        if (_weight_decay != 0)
            g_t += (_weight_decay * *p->data());
        if (_momentum != 0)
        {
            if (i > 1)
                _velocity[i] = _momentum * _velocity[i] + (1 - _dampening) * g_t;
            else
                _velocity[i] = g_t; // torch initializes velocity to the gradients instead of zeros
            if (_nestorov)
                g_t += _momentum * _velocity[i];
            else
                g_t = _velocity[i];
        }
        valarray<float> ds = (*p->data()) - _velocity[i];
        p->set_data(&ds);
    }
};

nn::Adam::Adam(std::vector<cyg::tptr<float>> parameters, float lr, float b1, float b2, float eps) : Optimizer(parameters), _lr(lr), _b1(b1), _b2(b2), _eps(eps)
{
    for (const auto &p : _parameters)
    {
        _velocity.push_back(valarray<float>(p->numel(), 0));
        _momentum.push_back(valarray<float>(p->numel(), 0));
    }
};

void nn::Adam::step()
{
    for (auto i = 0; i < _parameters.size(); i++)
    {
        auto p = _parameters[i];
        auto g = (*p->grad());
        _momentum[i] = _b1 * _momentum[i] + (1 - _b1) * g;
        _velocity[i] = _b2 * _velocity[i] + (1 - _b2) * pow(g, 2);
        auto m_ = _momentum[i] / (1 - pow(_b1, i + 1));
        auto v_ = _velocity[i] / (1 - pow(_b2, i + 1));
        valarray<float> dg = g - (_lr * m_) / (sqrt(v_) * _eps);
        p->set_data(&dg);
    }
};
inline cyg::tptr<float> nn::cross_entropy_loss(const cyg::tptr<float> logits, const cyg::tptr<int> target)
{
    if (logits->rank() != 2 || target->rank() != 1)
        throw std::runtime_error("invalid input, logits must be of rank 2 and targets must be 1D tensor");
    auto x_n = logits->at(target); // N
    auto logits_exp = logits->exp();
    auto out = x_n->exp() / (logits_exp->sum(-1) + 1e-20); // N / N => N
    out = -(out->log());                                   // N
    out = out->sum() / out->numel();
    out->grad_fn->name = "CrossEntropy";
    return out;
};

nn::Embedding::Embedding(const size_t &num_embeddings, const size_t &embedding_dim, const size_t &padding_idx, const bool &_freeze)
{}

cyg::tptr<float> nn::Embedding::forward(const cyg::tptr<int> &idx)
{
    return cyg::tptr<float>();
}
