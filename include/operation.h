#ifndef OPERATION_H
#define OPERATION_H
#include "util.h"
#include <map>
#include <valarray>
#include <vector>
#include <memory>

#include <iostream>
#include <numeric>
#include <random>


namespace cyg
{
    // template <typename T>
    //  void op_cpu(std::valarray<float>& out, const std::valarray<float>& lhs, const std::valarray<float>& rhs, T operation);

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
        }
    };

    template <class T>
    class Operation : public std::enable_shared_from_this<Operation<T>>
    {
    protected:
        std::unique_ptr<Context<T>> context;

    public:
        Operation()
        {
            this->context = std::make_unique<Context<T>>();
        };
        virtual void backward(T* incoming_grad) {};
        // Operation(const Operation &operation) : context(std::move(operation->context)) {} // rule of three/five/zero
        ~Operation() noexcept
        {
            this->context.reset();
        }
    };

    template <class T>
    class Add : public Operation<T>
    {
    public:
        Add() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs)
        {
            typename T::value_type dtype;
            assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
            if (lhs->get_device() != rhs->get_device())
                throw std::runtime_error("tensors are on different devices");
            auto req_grad = lhs->require_grad() || rhs->require_grad();
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>(); // allocating object on the heap,dont forget
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");
            *out_data = *lhs->data() + *rhs->data();
            auto out = std::make_shared<T>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
            if (lhs->require_grad() || rhs->require_grad())
                this->context->save_for_backward({lhs, rhs});

            return out;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            std::string err_msg = "cant backprop without executing a forward computation first";
            assertm(var.size() != 0, err_msg);
            if (var.size() == 0)
                throw std::runtime_error(err_msg);
            for (const auto &t : var)
                if (t->require_grad())
                {
                    // resize(t->n_elements(), incoming_grad->);
                    t->backward(incoming_grad);
                }
            this->context.reset();
        };
    };

    template <class T>
    class Mul : public Operation<T>
    {
    public:
        Mul() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs)
        {
            typename T::value_type dtype;
            assertm(lhs->get_device() == rhs->get_device(), "tensors are on different devices");
            if (lhs->get_device() != rhs->get_device())
                throw std::runtime_error("tensors are on different devices");
            auto req_grad = lhs->require_grad() || rhs->require_grad();
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>();
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");
            *out_data = *lhs->data() * *rhs->data();
            auto out = std::make_shared<T>(*out_data, lhs->shape(), lhs->get_device(), req_grad);
            this->context->save_for_backward({lhs, rhs});
            return out;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 2, "invalid");
            auto lhs = var[0];
            auto rhs = var[1];
            // resize(lhs->n_elements(), incoming_grad); // lhs same size with rhs
            if (rhs->require_grad())
            {
                std::valarray<float> data = *incoming_grad->data() * *lhs->data(); // y=a*b, dy/da = b = 1 * b
                auto grad = std::shared_ptr<T>(&data, rhs->shape(), incoming_grad->get_device(), false);
                rhs->backward(grad.get());
            }
            if (lhs->require_grad())
            {
                std::valarray<float> data = *incoming_grad->data() * *rhs->data();
                auto grad = std::shared_ptr<T>(&data, lhs->shape(), incoming_grad->get_device(), false);
                lhs->backward(grad.get());
            }
            this->context.reset();
        };
    };
    template <class T>
    class Div : public Operation<T>
    {
    public:
        Div() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &numerator, const std::shared_ptr<T> &denominator)
        {
            typename T::value_type dtype;
            assert(numerator->get_device() == denominator->get_device() && "tensors are on different devices");
            if (numerator->get_device() != denominator->get_device())
                throw std::runtime_error("tensors are on different devices");
            auto req_grad = numerator->require_grad() || denominator->require_grad();
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>();
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");
            *out_data = *numerator->data() / *denominator->data();
            auto out = std::make_shared<T>(*out_data, numerator->shape(), numerator->get_device(), req_grad);
            this->context->save_for_backward({numerator, denominator});

            return out;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 2, "invalid"); // for debugging purpose
            auto numerator = var[0];
            auto denominator = var[1];
            // resize(numerator->n_elements(), incoming_grad);
            // y= a/b y = a * b**-1   dy/da = b**-1=1/b  dy/db = -a*b**-2
            if (numerator->require_grad())
            {
                std::valarray<float> data = *incoming_grad->data() / *denominator->data();
                auto local_grad = std::make_shared<T>(&data, incoming_grad->shape(), incoming_grad->get_device(), false);
                numerator->backward(local_grad.get());
            }
            if (denominator->require_grad())
            { // dy/db = -a*b**-2
                std::valarray<float> data = *incoming_grad->data() * -1 * (*numerator->data() / std::pow(*denominator->data(), 2));
                auto local_grad = std::make_shared<T>(&data, incoming_grad->shape(), incoming_grad->get_device(), false);
                denominator->backward(local_grad.get());
            }
            this->context.reset();
        };
    };

    template <class T>
    class Pow : public Operation<T>
    {
    public:
        Pow() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &base, const std::shared_ptr<T> &exponent)
        {
            typename T::value_type dtype;
            assert(base->get_device() == exponent->get_device() && "tensors are on different devices");
            if (base->get_device() != exponent->get_device())
                throw std::runtime_error("tensors are on different devices");
            auto req_grad = base->require_grad() || exponent->require_grad();
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>();
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");
            *out_data = std::pow(*base->data(), *exponent->data());
            auto out = std::make_shared<T>(*out_data, base->shape(), base->get_device(), req_grad);
            this->context->save_for_backward({base, exponent});

            return out;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 2, "invalid"); // for debugging purpose
            auto base = var[0];
            auto exponent = var[1];
            // resize(base->n_elements(), incoming_grad);
            // y = b**e    dy/db = e*b**e-1
            if (base->require_grad())
            {
                std::valarray<float> data = *incoming_grad->data() * (*exponent->data() * std::pow(base->data(), exponent->data() - 1));
                auto local_grad = std::make_shared<T>(&data, incoming_grad->shape(), incoming_grad->get_device(), false);
                base->backward(local_grad.get());
            }
            if (exponent->require_grad())
            { // logy = elogb dy(1/y) = delogb dy/de = ylogb = b**e * logb     (logb natural log)
                std::valarray<float> data = *incoming_grad->data() * (std::pow(*base->data(), *exponent->data()) * std::log(*base->data()));
                auto local_grad = std::make_shared<T>(&data, incoming_grad->shape(), incoming_grad->get_device(), false);
                exponent->backward(local_grad.get());
            }
            this->context.reset();
        };
    };
    template <class T>
    class Mean : public Operation<T>
    {
    public:
        Mean() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &base, const int &dim, const bool &keepdim = true)
        {
            typename T::value_type dtype;
            CHECK_VALID_RANGE(dim, base->rank());
            std::valarray<size_t> strides, sizes, idxs;
            std::tie(strides, sizes, idxs) = generate_idxs(base->shape(), dim);
            std::valarray<decltype(dtype)> data = *base->data();
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>(idxs.size());
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");
            //@todo improve using gslice
            for (int i = 0; i < idxs.size(); i++)
            {
                auto m = std::valarray(data[std::slice(idxs[i], base->shape()[dim], strides[dim])]).sum() / base->shape()[dim];
                (*out_data)[i] = m;
            };
            std::vector<size_t> new_dims;
            new_dims.assign(begin(sizes), end(sizes));

            auto output = std::make_shared<T>(*out_data, new_dims, base->get_device(), base->require_grad());
            if (!keepdim)
                output->squeeze();

            this->context->save_for_backward({base});
            this->context->saved_data["dim"] = dim;

            return output;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid"); // for debugging purpose
            assertm(this->context->saved_data.size() == 2, "invalid");
            auto dim = this->context->saved_data["dim"];
            auto base = var[0];
            // y = (a_1 + a_2 + ... + a_n) / n    where n = base->shape()[dims]
            // dy/da_1 = 1/n
            // resize(base->n_elements(), incoming_grad);

            if (base->require_grad())
            {
                auto out_data = std::valarray<float>(base->n_elements(), 1 / base->shape()[dim]);
                std::valarray<float> data = *incoming_grad->data() * out_data;
                auto local_grad = std::make_shared<T>(&data, incoming_grad->shape(), incoming_grad->get_device(), false);
                base->backward(local_grad.get());
            }
            this->context.reset();
        };
    };
    template <class T>
    class Exp : public Operation<T>
    {
    public:
        Exp() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &exponent)
        {
            typename T::value_type dtype;
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>();
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");

            *out_data = std::exp(*exponent->data());

            auto out = std::make_shared<T>(*out_data, exponent->shape(), exponent->get_device(), exponent->require_grad());
            this->context->save_for_backward({exponent});

            return out;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid"); // for debugging purpose
            // y= e**x   logy = xloge   logy = x  i/y(dy) = dx  dy/dx = y = e**x
            auto base = var[0];
            // resize(base->n_elements(), incoming_grad);
            if (base->require_grad())
            {
                std::valarray<float> data = *incoming_grad->data() * std::exp(*base->data());
                auto local_grad = std::make_shared<T>(&data, incoming_grad->shape(), incoming_grad->get_device(), false);
                base->backward(local_grad.get());
            }
            this->context.reset();
        };
    };
    template <class T>
    class Log : public Operation<T>
    {
    public:
        Log() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &t)
        {
            typename T::value_type dtype;
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>();
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");

            *out_data = std::log(*t->data());

            auto out = std::make_shared<T>(*out_data, t->shape(), t->get_device(), t->require_grad());
            this->context->save_for_backward({t});
            // this->context->saved_data["base"] = base;

            return out;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid");
            auto base = var[0];
            // resize(base->n_elements(), incoming_grad);
            if (base->require_grad())
            { // y = logx   dy/dx = 1/x
                std::valarray<float> data = *incoming_grad->data() / *base->data();
                auto local_grad = std::make_shared<T>(&data, base->shape(), base->get_device(), false);
                base->backward(local_grad.get());
            }
            this->context.reset();
        };
    };

    template <class T>
    class Sum : public Operation<T>
    {
    public:
        Sum() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &base, const int &dim = -1, const bool &keepdim = true)
        {
            typename T::value_type dtype;
            CHECK_VALID_RANGE(dim, base->rank(), -1);
            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>(1);
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");
            std::vector<size_t> new_dims = {1};
            if (dim == -1)
                (*out_data)[0] = (*base->data()).sum();
            else
            {
                std::valarray<size_t> strides, sizes, idxs;
                std::tie(strides, sizes, idxs) = generate_idxs(base->shape(), dim);
                std::valarray<decltype(dtype)> data = *base->data();
                out_data->resize(idxs.size());
                //@todo improve using gslice
                for (int i = 0; i < idxs.size(); ++i)
                {
                    (*out_data)[i] = std::valarray(data[std::slice(idxs[i], base->shape()[dim], strides[dim])]).sum();
                };
                new_dims.assign(begin(sizes), end(sizes));
            }
            auto output = std::make_shared<T>(*out_data, new_dims, base->get_device(), base->require_grad());
            if (!keepdim && dim != -1)
                output->squeeze();

            this->context->save_for_backward({base});
            this->context->saved_data["dim"] = dim;
            return output;
        };
        void backward(T* incoming_grad) override
        {
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid");
            auto base = var[0];
            // y = a+b+c+d+.. dy/da = 1
            // resize(base->n_elements(), incoming_grad);
            if (base->require_grad())
            {
                auto out_data = std::valarray<float>(base->n_elements(), 1);
                std::valarray<float> data = *incoming_grad->data() * out_data;
                auto local_grad = std::make_shared<T>(&data, base->shape(), base->get_device(), false);
                base->backward(local_grad.get());
            }
            this->context.reset();
        };
    };


 template <class T>
    class Transpose : public Operation<T>
    {
    public:
        Transpose() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, int r=0, int c=1)
        {
            //only works for 0,1, 1,2. not working for 0,2 for now =fix
            typename T::value_type dtype;

            auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>(lhs->n_elements());
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");

            auto new_dims = lhs->shape();
            std::iter_swap(new_dims.begin()+r, new_dims.begin()+c);

            int n_ele = lhs->shape()[std::min(r, c)];
            
            std::valarray<size_t> col_strides, col_idxs, row_strides, row_idxs;
            std::tie(col_strides, std::ignore, col_idxs) = generate_idxs(new_dims, std::max(r,c)); //based on new dims
            std::tie(row_strides, std::ignore, row_idxs) = generate_idxs(lhs->shape(), std::min(r,c)); //based on old dims
            auto data = *lhs->data();
            // row_idxs.size() == col_idxs.size()
            for(int i=0;i<row_idxs.size(); i++){
                (*out_data)[std::slice(col_idxs[i] , n_ele, col_strides[std::max(r,c)])] = data[std::slice( row_idxs[i], n_ele, row_strides[std::min(r,c)])];
            }

            auto output = std::make_shared<T>(*out_data, new_dims, lhs->get_device(), lhs->require_grad());

            this->context->save_for_backward({lhs});

            return output;
        };

        void backward(T* incoming_grad) override
        {
            
            auto var = this->context->get_variables();
            assertm(var.size() == 1, "invalid");
            auto base = var[0];
            // resize(base->n_elements(), incoming_grad);
            if (base->require_grad())
            {
                auto out_data = std::valarray<float>(base->n_elements(), 1);
                std::valarray<float> data = *incoming_grad->data() * out_data;
                auto local_grad = std::make_shared<T>(data, base->shape(), base->get_device(), false);
                base->backward(local_grad.get());
            }
            this->context.reset();

        };
    };
    template<class T>
std::shared_ptr<T> matmul(T* lhs, T* rhs)
    {
        typename T::value_type dtype;
        const bool islhsgreater = lhs->rank()>=rhs->rank();
        std::vector<size_t> new_dims = islhsgreater ? lhs->shape() : rhs->shape();
        new_dims[islhsgreater ? new_dims.size()-1 : new_dims.size()-2 ] = islhsgreater ? rhs->shape()[rhs->rank() - 1] : lhs->shape()[lhs->rank() - 2];

        int n_elems = std::accumulate(new_dims.begin(), new_dims.end(), 1, std::multiplies<int>());

        auto out_data = new (std::nothrow) std::valarray<decltype(dtype)>(n_elems);
        if (out_data == nullptr)
            throw std::runtime_error("insufficient memory");
        auto req_grad = lhs->require_grad() || rhs->require_grad();            

        auto rhs_transpose = rhs->transpose(rhs->rank()-1, rhs->rank()-2);

        std::valarray<std::size_t> rhs_rows, rhs_strides, lhs_rows, lhs_strides;
        int lhs_dim = lhs->rank() - 1;
        int rhs_dim = rhs_transpose->rank() - 1;

        std::tie(lhs_strides, std::ignore, lhs_rows) =  generate_idxs(lhs->shape(), lhs_dim);
        std::tie(rhs_strides, std::ignore, rhs_rows) =  generate_idxs(rhs_transpose->shape(), rhs_dim);

        int n_ele = lhs->shape()[lhs_dim];

        auto ldata = *lhs->data();
        auto rdata = *rhs_transpose->data();
        
        int l_idx=0;
        for(int r_idx = 0; r_idx<n_elems; r_idx++){
            std::valarray<decltype(dtype)> l_slice = std::valarray(ldata[std::slice(lhs_rows[ l_idx % lhs_rows.size()], n_ele, lhs_strides[lhs_dim])]);
            std::valarray<decltype(dtype)> r_slice = std::valarray(rdata[std::slice(rhs_rows[ r_idx % rhs_rows.size()], n_ele, rhs_strides[rhs_dim])]);
            (*out_data)[r_idx] = (r_slice * l_slice).sum();
            l_idx += ((r_idx + 1)% rhs->shape()[rhs->rank()-1]==0);
        }
        auto output = std::make_shared<T>(*out_data, new_dims, lhs->get_device(), req_grad);

        return output;
    }
    template <class T>
    class MatMul : public Operation<T>
    {
    public:
        MatMul() : Operation<T>() {}
        std::shared_ptr<T> forward(const std::shared_ptr<T> &lhs, const std::shared_ptr<T> &rhs)
        {
            // ...*a*b   mm   ...*b*c = ...*a*c
            typename T::value_type dtype;
            
            auto out = matmul<T>(lhs.get(), rhs.get());

            this->context->save_for_backward({lhs, rhs});

            return out;
        };
        void backward(T* incoming_grad) override
        {
            // ...*a*b   mm   ...*b*c = ...*a*c
            //incominggrad = ...*a*c
            auto var = this->context->get_variables();
            assertm(var.size() == 2, "invalid");
            auto lhs = var[0];
            auto rhs = var[0];
            // y = a_11*b_11 + a_12*b_21 + a_13*b_31 + ...
            // dy/da_1* = b_11 + b_21 + b_31 + ...
            // dy/da = b.transpose() dy/db = a.transpose()
            // resize(lhs->n_elements(), incoming_grad);
            if (lhs->require_grad())
            {
                // resize(lhs->n_elements(), incoming_grad);
                auto local_grad =  matmul<T>(incoming_grad, rhs->transpose().get()); //transpose = ...*c*b
                lhs->backward(local_grad.get());
            }
            if (rhs->require_grad())
            {
                auto local_grad = matmul<T>(incoming_grad, lhs->transpose().get()); //transpose = ...*b*a
                rhs->backward(local_grad.get());
            }
            this->context.reset();
        };
    };

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
}
#endif