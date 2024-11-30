#ifndef OPERATION_H
#define OPERATION_H

#include <map>
#include <valarray>
#include <vector>
#include <memory>
#include "functional.h"
#include "util.h"
#include <iostream>
#include <numeric>
#include <random>

namespace cyg
{
    // template <typename T>
    //  void op_cpu(std::valarray<float>& out, const std::valarray<float>& lhs, const std::valarray<float>& rhs, T operation);

    // TODO use heterogeneous containers for the cache to save different tensor types
    // boost.any comes to the rescue


    template <class T>
    class Context
    {
        std::vector<std::shared_ptr<T>> cache; // check T destructor, vector deallocates,
    public:
        Context() {}
        std::map<std::string, int> saved_data;

        /**
         * @brief Save input tensors[shared pointers] that can be retrieved later.
         * the Ts can be retrieved by calling the get_saved_variables() method
         * for ex.
         *  const vector<int>dims{1,2,3};
         *  auto arg1 = ones(dims);
         *  auto arg2 = ones(dims);
         *  ctx->save_for_backward({arg1, arg2});
         *
         * @param Ts(type std::vector<std::shared_ptr<T>>)
         */
        void save_for_backward(std::vector<std::shared_ptr<T>> Ts)
        {
            for (const auto &t : Ts)
                this->cache.push_back(t);
        };
        /**
         * @brief Retrieve tensors that have been saved previously.
         * for ex.
         *  const vector<int>dims{1,2,3};
         *  auto arg1 = ones(dims);
         *  auto arg2 = ones(dims);
         *  ctx->save_for_backward({arg1, arg2});
         *  ctx->get_variables()=={arg1, arg2};
         *
         * @return the vectors of Ts shared_pointers (type std::vector<std::shared_ptr<tensor>>)
         */
        std::vector<std::shared_ptr<T>> get_variables() { return this->cache; };
        // Context(const Context<T> &context) : cache(context->cache), saved_data(context->saved_data) {} // rule of three/five/zero
        ~Context() noexcept
        {
            this->saved_data.clear();
            this->cache.clear();
        }
    };

    template <class T>
    class Operation : public std::enable_shared_from_this<Operation<T>>
    {
    protected:
        std::unique_ptr<Context<T>> context;

    public:
        bool _done = false; //flag to indicate when you backprop through all descendant nodes (children)
        Operation()
        {
            this->context = std::make_unique<Context<T>>();
        };
        std::string name;
        void reset(){
            this->context.reset(new Context<T>());
        }
        void backward(std::shared_ptr<T> incoming_grad){
            if(_done) {
                std::cout<<"trying to backprop on this node again, pls be sure this is intended"<<"\n";
                return;
            }
            return _backward(incoming_grad);
        }
        virtual void _backward(std::shared_ptr<T> incoming_grad) {};
        // Operation(const Operation &operation) : context(std::move(operation->context)) {} // rule of three/five/zero
        friend std::ostream& operator<<(std::ostream& out, const Operation& op){
            out<<op.name<<"Op";
            return out;
        }
        ~Operation() noexcept
        {
            this->context.reset(new Context<T>());
        }
    };

    template <class T>
    class Add : public Operation<T>    {
    public:
        Add() : Operation<T>() { this->name="Add";}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs)
        {
            auto output = functional::add(*lhs, *rhs);
            if(output->requires_grad()) this->context->save_for_backward({lhs, rhs});
            return output;
        };
        void _backward(std::shared_ptr<T> incoming_gradient)
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD<T>(var, 2);
            for (const auto &t : var)
                if (t->requires_grad()) 
                { 
                    auto cloned_grad = incoming_gradient->clone(false);
                    cloned_grad->sum_to_size(t->shape());
                    t->backward(cloned_grad);
                }
            this->_done = true;;
        };
    };

    template <class T>
    class Mul : public Operation<T>
    {
    public:
        Mul() : Operation<T>() { this->name="Mul";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs)
        {
            auto output = functional::mul(*lhs, *rhs);
            if(output->requires_grad()) this->context->save_for_backward({lhs, rhs});

            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD<T>(var, 2);
            auto lhs = var[0];
            auto rhs = var[1];
            if (rhs->requires_grad())
            {
                auto local_grad = functional::mul(*incoming_grad, *lhs);
                local_grad->requires_grad_(false);
                local_grad->sum_to_size(rhs->shape());
                rhs->backward(local_grad);
            }
            if (lhs->requires_grad())
            {
                auto local_grad = functional::mul(*incoming_grad, *rhs);
                local_grad->requires_grad_(false);
                local_grad->sum_to_size(lhs->shape());              
                lhs->backward(local_grad);
            }
            this->_done = true;;
        };
    };
    template <class T>
    class Div : public Operation<T>
    {
    public:
        Div() : Operation<T>() {this->name="Div";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& numerator, const std::shared_ptr<T>& denominator)
        {
            auto output = functional::div(*numerator, *denominator);
            if(output->requires_grad()) this->context->save_for_backward({numerator, denominator});

            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD<T>(var, 2);
            auto numerator = var[0];
            auto denominator = var[1];
            // y= a/b y = a * b**-1   dy/da = b**-1=1/b  dy/db = -a*b**-2
            if (numerator->requires_grad())
            {
                auto local_grad = functional::div(*incoming_grad, *denominator);
                local_grad->requires_grad_(false);
                local_grad->sum_to_size(numerator->shape());
                numerator->backward(local_grad);
            }
            if (denominator->requires_grad())
            { // dy/db = -a*b**-2
                auto cn = -(numerator->clone(false));
                auto cd = denominator->clone(false)->pow(2);
                auto local_grad = functional::mul(*incoming_grad, *functional::div(*cn, *cd));
                local_grad->sum_to_size(denominator->shape());
                denominator->backward(local_grad);
            }
            this->_done = true;;
        };
    };

    template <class T>
    class Pow : public Operation<T>
    {
    public:
        Pow() : Operation<T>() {this->name="Pow";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& base, const std::shared_ptr<T>& exponent)
        {
            auto output = functional::pow(*base, *exponent);
            if(output->requires_grad()) this->context->save_for_backward({base, exponent});

            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD<T>(var, 2);
            auto base = var[0];
            auto exponent = var[1];
            // y = b**e    dy/db = e*b**e-1
            if (base->requires_grad())
            {
                auto cloned_grad = incoming_grad->clone(false);
                cloned_grad->sum_to_size(base->shape());
                auto cloned_exponent = exponent->clone(false);
                auto cloned_base = base->clone(false);
                auto eb = functional::mul(*cloned_exponent, *(cloned_base->pow(cloned_exponent - 1)));
                auto local_grad = functional::mul(*cloned_grad, *eb);
                base->backward(local_grad);
            }
            if (exponent->requires_grad())
            { // logy = elogb dy(1/y) = delogb dy/de = ylogb = b**e * logb     (logb natural log)
                auto cloned_grad = incoming_grad->clone(false);
                cloned_grad->sum_to_size(exponent->shape());
                auto cloned_base = base->clone(false);
                auto be = functional::mul(*functional::pow(*cloned_base, *exponent->clone(false)), *(functional::log(*cloned_base)));
                auto local_grad = functional::mul(*cloned_grad, *be);
                exponent->backward(local_grad);
            }
            this->_done = true;;
        };
    };
    
    template <class T>
    class Sum : public Operation<T>
    {
    public:
        Sum() : Operation<T>() {this->name="Sum";}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &base, int dim = INT_MAX, const bool& keepdim = false)
        {
            auto output = functional::sum(*base, dim, keepdim);
            if(output->requires_grad()) {
                this->context->save_for_backward({base});
                this->context->saved_data["dim"] = dim<0? base->rank() + dim : dim;
                this->context->saved_data["keepdim"] = keepdim;
            }
            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD(var, 1);
            auto base = var[0];
            auto dim = this->context->saved_data["dim"];
            auto keepdim = this->context->saved_data["keepdim"];
            // y = a+b+c+d+.. dy/da = 1
            if (base->requires_grad())
            {
                auto cloned_grad = incoming_grad->clone(false);
                if(dim!=INT_MAX && !keepdim) cloned_grad->unsqueeze(dim);
                cloned_grad->expand(base->shape());
                cloned_grad->sum_to_size(base->shape());
                base->backward(cloned_grad);
            }
            this->_done = true;;
        };
    };

    template <class T>
    class Mean : public Operation<T>
    {
    public:
        Mean() : Operation<T>() {this->name="Mean";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& base, int dim = INT_MAX, const bool &keepdim = true)
        {
            auto output = functional::mean(*base, dim, keepdim);
            if(output->requires_grad()){
                this->context->save_for_backward({base});
                this->context->saved_data["dim"] =  dim<0? base->rank() + dim : dim;
                this->context->saved_data["keepdim"] = keepdim;
            }
            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD<T>(var, 1);
            assertm(this->context->saved_data.size() == 1, "invalid");
            auto base = var[0];
            auto dim = this->context->saved_data["dim"];
            auto keepdim = this->context->saved_data["keepdim"];
            // y = (a_1 + a_2 + ... + a_n) / n    where n = base->shape()[dims]
            // dy/da_1 = 1/n
            if (base->requires_grad())
            {
                auto cloned_grad = incoming_grad->clone(false);
                cloned_grad/=(dim==INT_MAX? base->numel() : base->shape()[dim]);                

                if(dim!=INT_MAX && !keepdim) cloned_grad->unsqueeze(dim);
                cloned_grad->expand(base->shape());
                
                // cloned_grad->sum_to_size(base->shape());
                base->backward(cloned_grad);
            }
            this->_done = true;;
        };
    };

    template <class T>
    class Exp : public Operation<T>
    {
    public:
        Exp() : Operation<T>() {this->name="Exp";}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &base)
        {
            auto output = functional::exp(*base);
            if(output->requires_grad()) this->context->save_for_backward({base});

            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid"); // for debugging purpose
            // y= e**x   logy = xloge   logy = x  i/y(dy) = dx  dy/dx = y = e**x
            auto base = var[0];
            if (base->requires_grad())
            {
                auto local_grad = functional::mul(*incoming_grad, *functional::exp(*base));
                local_grad->requires_grad_(true);
                base->backward(local_grad);
            }
            this->_done = true;;
        };
    };
    template <class T>
    class Log : public Operation<T>
    {
    public:
        Log() : Operation<T>() {this->name="Log";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& base)
        {
            auto output = functional::log(*base);
            if(output->requires_grad()) this->context->save_for_backward({base});
            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid");
            auto base = var[0];

            if (base->requires_grad())
            { // y = logx   dy/dx = 1/x
                auto cloned_base = base->clone(false);
                auto local_grad = incoming_grad / cloned_base;
                base->backward(local_grad);
            }
            this->_done = true;;
        };
    };

    template <class T>
    class Transpose : public Operation<T>
    {
    public:
        Transpose() : Operation<T>() {this->name="Transpose";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, int d1 = -1, int d2 = -2)
        {
            // only works for 0,1, 1,2. not working for 0,2 for now =fix
            auto output = functional::transpose(*lhs, d1, d2);
            if(output->requires_grad()){
                this->context->save_for_backward({lhs});
                this->context->saved_data["d1"] = d1;
                this->context->saved_data["d2"]=d2;
            }
            return output;
        };

        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD(var, 1);
            auto base = var[0];
            auto d1 = this->context->saved_data["d1"];
            auto d2 = this->context->saved_data["d2"];

            if (base->requires_grad())
            { 
                auto cloned_grad = incoming_grad->clone(false);
                cloned_grad->transpose(d1, d2, true);
                base->backward(cloned_grad);
            }
            this->_done = true;;
        };
    };

    template <class T>
    class Var : public Operation<T>
    {
    public:
        Var() : Operation<T>() {this->name="Var";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& base, int dim = INT_MAX, const int& correction =1 ,const bool& keepdim = true)
        {
            auto output = functional::var(*base, dim, correction, keepdim);
            if(output->requires_grad()){
                this->context->save_for_backward({base});
                this->context->saved_data["dim"] = dim<0? base->rank() + dim : dim;
                this->context->saved_data["keepdim"] = keepdim;
                this->context->saved_data["correction"] = correction;
            }
            return output;
        };

        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD(var, 1);
            auto base = var[0];
            auto dim = this->context->saved_data["dim"];
            auto keepdim = this->context->saved_data["keepdim"];
            auto correction = this->context->saved_data["correction"];
            // y = (  ((x_1 - x_)^2 + (x_2 - x_)^2) / n
            // dy/dx_1 = 2(x_1 - x) / (n-1) = 2(x_1 - x) / n * = 2 * (x - x_) / n-1
            if (base->requires_grad())
            {
                auto cloned_base = base->clone(false);
                auto local_grad = incoming_grad->clone(false);

                if(!keepdim && dim!=INT_MAX) local_grad->unsqueeze(dim);
                local_grad->expand(base->shape());

                int n_elements = dim==INT_MAX? base->numel() : base->shape()[dim];

                auto mean = cloned_base->mean(dim, true);
                mean->expand(base->shape());
                auto x_x_ = cloned_base - mean;

                auto grad = (local_grad * 2 * x_x_) / std::max(0, n_elements - correction);
                // grad->sum_to_size(base->shape());
                base->backward(grad);
            }
            this->_done = true;;
        };
    };
    
    template <class T>
    class MatMul : public Operation<T>
    {
    public:
        MatMul() : Operation<T>() {this->name="MatMul";}
        std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs)
        {
            // ...*a*b   mm   ...*b*c = ...*a*c
            //@todo clean up matmul and MatMul forward and backward
            auto output = functional::matmul(*lhs, *rhs);
            if(output->requires_grad()) this->context->save_for_backward({lhs, rhs});

            return output;
        };
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            // ...*a*b   mm   ...*b*c = ...*a*c
            // incominggrad = ...*a*c
            auto var = this->context->get_variables();
            CHECK_BACKWARD(var, 2);
            auto lhs = var[0];
            auto rhs = var[1];
            // y = a_11*b_11 + a_12*b_21 + a_13*b_31 + ...
            // dy/da_1* = b_11 + b_21 + b_31 + ...
            // dy/da = b.transpose() dy/db = a.transpose()
            if (lhs->requires_grad())
            {
                auto cloned_rhs = rhs->clone(false);
                cloned_rhs->transpose(-1, -2, true);
                auto out_tensor = functional::matmul(*incoming_grad, *cloned_rhs); // transpose = ...*c*b
                out_tensor->sum_to_size(lhs->shape());
                lhs->backward(out_tensor);
            }
            if (rhs->requires_grad())
            {
                auto cloned_lhs = lhs->clone(false);
                cloned_lhs->transpose(-1, -2, true);
                auto out_tensor = functional::matmul(*cloned_lhs, *incoming_grad); // transpose = ...*b*a
                out_tensor->sum_to_size(rhs->shape());
                rhs->backward(out_tensor);
            }
            this->_done = true;;
        };
    };

    template<class T>
    class Mask : public Operation<T>
    {
        public:
            Mask() : Operation<T>() {this->name="Mask";}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& condition, const std::shared_ptr<T>& true_value, const std::shared_ptr<T>& false_value)
            {
                auto output = functional::mask(*condition, *true_value, *false_value);
                if(output->requires_grad()) this->context->save_for_backward({true_value, false_value, condition});
                return output;
            } 
        void _backward(std::shared_ptr<T> incoming_grad) override
        {
            this->_done = false;
            auto var = this->context->get_variables();
            CHECK_BACKWARD(var, 3); //should be 3
            auto true_value = var[0];
            auto false_value = var[1];
            auto condition = var[2];
            if(true_value->requires_grad()){
                auto local_grad = incoming_grad->clone(false);
                (*local_grad->data())[(*condition->data())<=0] = 0;
                true_value->backward(local_grad);
            }

            if(false_value->requires_grad()){
                auto local_grad = incoming_grad->clone(false);
                (*local_grad->data())[(*condition->data())>0] = 0;
                false_value->backward(local_grad);
            }
            this->_done = true;;   
        }
    };
    // TODO context's cache with heterogenous containers
    template<class T>
    class Slice : public Operation<T>
    {
        public:
            Slice() : Operation<T>() {this->name="Slice";}
            std::shared_ptr<T> forward(const std::shared_ptr<T> &t, const std::shared_ptr<T> &indices, int dim=-1) //temporarily cast indices to float, fix with herogenous containers
            {
                auto output = functional::slice<float, int>(*t, *indices, dim);
                if(output->requires_grad()) {
                    this->context->save_for_backward({t, indices});
                    this->context->saved_data["dim"] = dim<0? t->rank()+dim: dim;
                }
                return output;
            } 
            void _backward(std::shared_ptr<T> incoming_grad) override
            {
                this->_done = false;
                auto var = this->context->get_variables();
                CHECK_BACKWARD(var, 2);
                auto t = var[0];
                auto indices = var[1];
                auto dim = this->context->saved_data["dim"];
                // incoming_grad = N along dim of t
                // t = *, N
                if(t->requires_grad()){
                    auto local_grad = t->clone(false, 0); // N*a..
                    // auto new_d =  new std::valarray<T>(0, t->numel());
                    // auto sh = t->shape();
                    // sh[dim]=1; auto fac = std::accumulate(sh.begin(), sh.end(), 1, std::multiplies<int>());
                    // // assertm(idxs.size()==incoming_grad->numel())
                    // / ERROR fix implementation here
                    std::valarray<size_t> dfs = {0, 2, 1,2, 4,4,5,5};
                    std::valarray<float> res = std::valarray((*t->data())[dfs]);
                    std::cout<<res.size()<<"\n";
                    // local_grad->set_data(new_d);
                    t->backward(local_grad);
                }
                
                this->_done = true;;   
            }
    };

    // template<class T>
    // class Stack : public Operation<T>
    // {
    //     public:
    //         Stack() : Operation<T>() {this->name="Stack";}
    //         std::shared_ptr<T> forward(const std::vector<std::shared_ptr<T>> ts, int dim=0)
    //         {
    //             auto output = functional::stack(ts, dim);
    //             if(output->requires_grad()) this->context->save_for_backward(ts);
    //             return output;
    //         } 
    //     void backward(std::shared_ptr<T> incoming_grad) override
    //     {
    //         auto var = this->context->get_variables();
    //         // CHECK_BACKWARD(var, 3); //should be 3
            
    //         this->reset();   
    //     }
    // };
    // template <class T>
    // class Cos : public Operation<T>
    // {
    // public:
    //     Cos() : Operation<T>() {}
    //     std::shared_ptr<T> forward(const std::shared_ptr<T>& v1);
    //     void backward(std::valarray<float> *incoming_grad) override;
    // };

    // template <class T>
    // class Sin : public Operation<T>
    // {
    // public:
    //     Sin() : Operation<T>() {}
    //     std::shared_ptr<T> forward(const std::shared_ptr<T>& v1);
    //     void backward(std::valarray<float> *incoming_grad) override;
    // };
};
#endif