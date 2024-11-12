#ifndef TENSOR_H
#define TENSOR_H

#include "utils.h"
#include "operation.h"
#include <valarray>
#include <array>
#include <new>
#include <iostream>
#include <memory>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <sstream>
#include <type_traits>

// TODO integrate cuda - thrust - focus on tensor.d and tensor.grad

namespace cyg
{

    template <class T>
    class tensor : public std::enable_shared_from_this<tensor<T>>
    {
    public:
        typedef T value_type;
        std::shared_ptr<Operation<tensor<T>>> grad_fn;
        tensor(std::vector<size_t> dims, int value = 0, bool requires_grad = false) : d(initialize<T>(dims, value)), dims(dims), requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            if (requires_grad)
            {
                if (typeid(T) != typeid(float &))
                    throw std::runtime_error(ERROR_GRAD_DTYPE);
                this->zero_grad();
            }
        }

        explicit tensor(std::vector<size_t> dims, std::valarray<T> *data = nullptr, bool requires_grad = false) : dims(dims), requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            delete this->d;
            if (data != nullptr)
            {
                CHECK_SIZE(dims, data->size());
                this->d = data;
            }
            else
                this->d = initialize<T>(dims, 0);
            if (requires_grad)
            {
                if (typeid(T) != typeid(float &))
                    throw std::runtime_error(ERROR_GRAD_DTYPE);
                this->zero_grad();
            }
        };
        /**
         * @brief class method to turn on/off grad for the given tensor
         *
         * @param requires_grad(type bool)
         */
        void enable_grad(bool requires_grad)
        {
            if (requires_grad)
            {
                this->requires_grad = true;
                this->zero_grad();
            }
            else
            {
                this->requires_grad = false;
                delete this->grad;
                this->grad = nullptr;
            };
        };
        // // tensor(const tensor& other):grad_fn(other.grad_fn), d(other.d), grad(other.grad), requires_grad(other.requires_grad), _prev(other._prev),dims(other.dims){}
        // ~tensor() noexcept { delete d; }; // destructor must not fail - so noexcept
        std::valarray<T> *data() const { return this->d; };
        std::vector<size_t> shape() const { return this->dims; };
        const int n_elements() const { return this->d->size(); };
        const int rank() const { return this->dims.size(); };
        std::valarray<float> *get_grad()
        {
            if (this->grad_fn != nullptr)
                throw std::runtime_error(WARNING_GRAD_NOT_LEAF);
            return this->grad;
        };
        const bool require_grad() const { return this->requires_grad; };
        void zero_grad()
        {
            delete this->grad;
            this->grad = initialize<float>(this->dims, 0);
        };
        /**
         * @brief remove dim with size of 1 from the given tensor
         *
         */
        void squeeze()
        {
            this->dims.erase(std::remove(this->dims.begin(), this->dims.end(), 1), this->dims.end());
        };
        /**
         * @brief add dim of size 1 at the input dim of the given tensor
         *
         * @param dim(type int)
         */
        std::shared_ptr<tensor<T>> unsqueeze(int dim)
        {
            CHECK_VALID_RANGE(dim, this->rank() + 1, -this->rank() - 1);
            this->dims.insert(this->dims.begin() + (dim < 0 ? dim + this->rank() + 1 : dim), 1);
            return this->shared_from_this();
        };
        /**
         * @brief class method to backprop on given tensor
         *
         * @param incoming_gradient(type std::vector<float>)
         */
        void backward(tensor<T> *incoming_gradient = nullptr)
        {
            // assertm(this->requires_grad, "please enable requires_grad on tensor to backprop");
            if (incoming_gradient == nullptr && this->n_elements() != 1)
                throw std::runtime_error(ERROR_NON_SCALAR_BACKPROP);

            if (incoming_gradient == nullptr)
                incoming_gradient = make_shared<tensor<T>>(this->dims, 1).get(); // scalar, size=1
            if (incoming_gradient->n_elements() != this->n_elements())
                throw std::runtime_error(ERROR_GRAD_MISMATCH);

            if (this->grad != nullptr)
            {
                *this->grad += *incoming_gradient->data();
            }
            if (this->grad_fn != nullptr)
                this->grad_fn->backward(incoming_gradient);
        };
        /**
         * @brief operator overloading for the function call to index a tensor
         *
         * @return value at the input index (type float)
         */
        T operator()(size_t d, ...)
        {
            std::vector<size_t> dims = {d};
            va_list args;
            va_start(args, d);
            for (int i = 0; i < this->rank() - 1; ++i)
            {
                auto dd = va_arg(args, size_t);
                dims.push_back(dd);
            }
            va_end(args);
            CHECK_VALID_INDEX(dims, this->dims);
            auto index = get_index(&this->dims, dims);
            return (*this->d)[index];
        };
        T operator[](size_t dims); // index a 1D tensor
        // std::valarray<T> operator()(int dims, ...);

        /**
         * @brief addition operation for tensors, tensors must have equal shape
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        std::shared_ptr<tensor<T>> add(const std::shared_ptr<tensor<T>> other)
        {
            CHECK_ARGS_OPS(this->shape(), other->shape());
            auto add_op = std::make_shared<cyg::Add<tensor<T>>>();
            auto output = add_op->forward(this->shared_from_this(), other);
            if (this->requires_grad)
                output->grad_fn = add_op;
            return output;
        };
        /**
         * @brief element wise multiplication operation for tensors, tensors must have equal shape
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        std::shared_ptr<tensor<T>> mul(const std::shared_ptr<tensor<T>> other)
        {
            CHECK_ARGS_OPS(this->shape(), other->shape());
            auto mul_op = std::make_shared<Mul<tensor<T>>>();
            auto output = mul_op->forward(this->shared_from_this(), other);
            if (this->requires_grad)
                output->grad_fn = mul_op;
            return output;
        };
        std::shared_ptr<tensor<T>> mm(const std::shared_ptr<tensor<T>> other)
        {
            CHECK_MM_DIMS(this->shape(), other->shape());
            auto mat_mul_op = std::make_shared<MatMul<tensor<T>>>();
            auto output = mat_mul_op->forward(this->shared_from_this(), other);
            if (this->requires_grad)
                output->grad_fn = mat_mul_op;

            return output;
        };

        /**
         * @brief element wise division operation for tensors, tensors must have equal shape
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        std::shared_ptr<tensor<T>> div(const std::shared_ptr<tensor<T>> other)
        {
            CHECK_ARGS_OPS(this->shape(), other->shape());
            auto div_op = std::make_shared<Div<tensor<T>>>();
            auto output = div_op->forward(this->shared_from_this(), other);
            if (this->requires_grad)
                output->grad_fn = div_op;
            return output;
        };
        /**
         * @brief element wise exponent operation for tensors, tensors must have equal shape
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        std::shared_ptr<tensor<T>> pow(const std::shared_ptr<tensor<T>> exponent, const bool &inplace = false)
        {
            CHECK_ARGS_OPS(this->shape(), exponent->shape());
            auto pow_op = std::make_shared<Pow<tensor<T>>>();
            auto output = pow_op->forward(this->shared_from_this(), exponent);
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = pow_op;
            return output;
        };

        /**
         * @brief scalar exponent operation for tensors, raise a tensor to a scalar power
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        std::shared_ptr<tensor<T>> pow(const float &exponent, const bool &inplace = false) //@todo use template
        {
            const auto t = std::make_shared<tensor<T>>(this->dims, exponent, this->requires_grad); //@todo use named args for constructor
            return this->pow(t, inplace);
        }

        /**
         * @brief compute the exponent of a tensor along
         * for ex
         *
         *
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        std::shared_ptr<tensor<T>> exp(const bool &inplace = false)
        {
            auto exp_op = std::make_shared<Exp<tensor<T>>>();
            auto output = exp_op->forward(this->shared_from_this());
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = exp_op;
            return output;
        };
        /**
         * @brief compute the natural log of a tensor along
         * for ex
         *
         *
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        std::shared_ptr<tensor<T>> log(const bool &inplace = false)
        {
            auto log_op = std::make_shared<Log<tensor<T>>>();
            auto output = log_op->forward(this->shared_from_this());
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = log_op;
            return output;
        };

        /**
         * @brief compute the mean along the input dimension for a given tensor
         * for ex
         *  auto t = cyg::ones({2,3,4})
         *  t.mean(2) // mean along the dimension of size 4
         *
         * @param dims(type: int)
         * @param keepdims(type: bool)
         *
         * @return tensor (type cyg::tensor)
         */
        std::shared_ptr<tensor<T>> mean(int dim = INT_MAX, const bool &inplace = false, const bool &keepdims = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            auto mean_op = std::make_shared<Mean<tensor<T>>>();
            auto output = mean_op->forward(this->shared_from_this(), dims, keepdims);
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = mean_op;
            return output;
        };

        /**
         * @brief compute the sum of each row of the given tensor in the input dimension
         * for ex
         *
         * @param dim(type int)
         * @param keepdim(type bool)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        std::shared_ptr<tensor<T>> sum(int dim = INT_MAX, const bool &inplace = false, const bool &keepdim = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            auto sum_op = std::make_shared<Sum<tensor<T>>>();
            auto output = sum_op->forward(this->shared_from_this(), dim, keepdim);
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = sum_op;
            return output;
        };
        std::shared_ptr<tensor<int>> argmax(int dim = INT_MAX, const bool &keepdim = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            std::vector<size_t> new_dims;
            std::valarray<int> id_data;

            auto data = (*this->d);
            auto out_data = new (std::nothrow) std::valarray<int>(1);
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");

            if (dim == INT_MAX) // flattened input
            {
                id_data.resize(this->n_elements());
                std::iota(std::begin(id_data), std::end(id_data), 0);
                (*out_data)[0] = std::valarray(id_data[data == data.max()])[0];
                new_dims = {1};
            }
            else
            {
                if (dim < 0)
                    dim = this->rank() + dim;
                id_data.resize(this->dims[dim]);
                std::iota(std::begin(id_data), std::end(id_data), 0);

                const auto [strides, start_idxs] = generate_idxs(this->shape(), dim);

                out_data->resize(start_idxs.size());
                for (auto i = 0; const auto &idx : start_idxs)
                {
                    auto gslice = std::slice(idx, this->shape()[dim], strides[dim]);
                    auto data_slice = std::valarray(data[gslice]);
                    (*out_data)[i++] = std::valarray(id_data[data_slice == data_slice.max()])[0];
                }
                new_dims = this->shape();
                new_dims[dim] = 1;
            }
            auto output = std::make_shared<tensor<int>>(new_dims, out_data, false);
            if (!keepdim && dim != INT_MAX)
                output->squeeze();
            return output;
        };
        void triu() // inplace op
        {
            // number of elements below main diagonal ==
            const auto [_, idxs] = generate_idxs(this->shape(), -1);
            int itr = 1;
            for (int i = 1; i < idxs.size(); ++i)
            {
                itr %= this->shape()[this->rank() - 2];
                auto gsl = std::slice(idxs[i], itr, 1);
                (*this->d)[gsl] = 0;
                if (this->requires_grad)
                    (*this->grad)[gsl] = 0;
                itr++;
            }
        };
        // main diagonal for now i.e diagonal=0;
        std::shared_ptr<tensor<T>> transpose(int d1 = -1, int d2 = -2, const bool inplace = false)
        {
            CHECK_TRANSPOSE(this->shape(), d1, d2);
            auto tr_op = std::make_shared<Transpose<tensor<T>>>();
            auto output = tr_op->forward(this->shared_from_this(), d1, d2);
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = tr_op;
            return output;
        };

        std::shared_ptr<tensor<T>> var(int dim = INT_MAX, const bool &inplace = false, const bool &keepdim = true)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            auto var_op = std::make_shared<Var<tensor<T>>>();
            auto output = var_op->forward(this->shared_from_this(), dim, keepdim);
            if (inplace && !this->requires_grad)
            {
                delete this->d;
                this->d = output->data();
                this->dims = output->shape();
                return this->shared_from_this();
            }
            if (this->requires_grad)
                output->grad_fn = var_op;
            return output;
        }

        void uniform(const float &low, const float &high)
        { // inplace
            std::generate(std::begin((*this->d)), std::end((*this->d)), [low, high]()
                          { return generate_random(low, high); });
        }
        void repeat(int dim, const int &n_repeat)
        { // in place op
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            if (dim < 0)
                dim = this->rank() + dim;
            this->dims[dim] = n_repeat;

            auto new_d = repeat_ND<T>(this->d, this->dims, dim, n_repeat);
            delete this->d;
            this->d = new_d;
            if (this->requires_grad){
                auto new_grad = repeat_ND<float>(this->grad, this->dims, dim, n_repeat);
                delete this->grad;
                this->grad = new_grad;             }
        }

        // tensor(tensor&& other);
        // tensor& operator=(tensor&& other);
        // disable copy constructor and copy assigment operator
        // tensor(const tensor&);
        // tensor& operator=(const tensor&);

        std::valarray<T> *d = nullptr;
        //@todo use smart ptr for d and grad
        std::vector<size_t> dims;
        std::valarray<float> *grad = nullptr;
        bool requires_grad;
    };

    /**
     * @brief create a tensor with randomly generated data from a uniform distribution
     * @ todo add a param for a generator (different distributions)
     *
     * @param dims(type std::vector<int>)
     * @param low(type int)
     * @param high(type int)
     * @param requires_grad(type bool)
     *
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    std::shared_ptr<tensor<float>> randn(std::vector<size_t> dims, int low = -1, int high = 1, bool requires_grad = false)
    {
        assertm(low < high, "low must be lower than high, pls check your input params");
        if (low >= high)
            throw std::runtime_error("pls check input params, the value for the low arg must be lower than the high args");
        auto vec = initialize<float>(dims, 1);
        std::generate(std::begin(*vec), std::end(*vec), [low, high]()
                      { return generate_random(low, high); });
        return std::make_shared<tensor<float>>(dims, vec, requires_grad);
    };

    /**
     * @brief overloading cout operator for tensor
     *
     * @param out(type std::ostream)
     * @param input_tensor(type cyg::tensor)
     * @return output_stream(type std::ostream)
     */
    template <class T>
    std::ostream &operator<<(std::ostream &out, const tensor<T> &t)
    {
        std::valarray<T> data = (*t.data());
        auto shape = t.shape();
        out << "(";
        std::stringstream output_string = printND<T>(data, shape);
        out << output_string.str() << ", size = ( ";
        for (const auto &r : t.shape())
            out << r << " ";
        out << "), requires_grad = " << std::boolalpha << t.require_grad() << " )" << "\n";
        return out;
    };

    template <class T>
    std::ostream &operator<<(std::ostream &out, const std::vector<T> input)
    {
        out << "(";
        for (int i = 0; i < input.size() - 1; i++)
            out << input[i] << " , ";
        out << input[input.size() - 1];
        out << ")";
        return out;
    };

    /**
     * @brief create a tensor with 1s and same shape as input tensor
     *
     *
     * @param input_tensor(type std::shared_ptr<tensor>)
     * @param requires_grad(type bool)
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    template <class T>
    std::shared_ptr<tensor<T>> ones_like(const std::shared_ptr<tensor<T>> &input_tensor, bool requires_grad = false)
    {
        return std::make_shared<tensor<T>>(input_tensor->shape(), 1, requires_grad);
    };
    /**
     * @brief create a tensor with 0s and same shape as input tensor
     *
     *
     * @param input_tensor(type std::shared_ptr<tensor>)
     * @param requires_grad(type bool)
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    template <class T>
    std::shared_ptr<tensor<T>> zeros_like(const std::shared_ptr<tensor<T>> &input_tensor, bool requires_grad = false)
    {
        return std::make_shared<tensor<T>>(input_tensor->shape(), 0, requires_grad);
    };

    template <class T>
    std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> rhs) { return lhs->add(rhs); };
    template <class T>
    std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> lhs, const float &rhs) { return lhs + std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false); };
    template <class T>
    std::shared_ptr<tensor<T>> operator+=(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs + rhs;
    };
    template <class T>
    std::shared_ptr<tensor<T>> operator+=(std::shared_ptr<tensor<T>> lhs, const float &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs + rhs;
    };

    /**
     * @brief scalar multiplication operation for tensors, multiply a tensor by a scalar
     *
     * @param other(type: float)
     * @return tensor (type cyg::tensor)
     */
    template <class T>
    std::shared_ptr<tensor<T>> operator*(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> rhs) { return lhs->mul(rhs); };
    template <class T>
    std::shared_ptr<tensor<T>> operator*(std::shared_ptr<tensor<T>> lhs, const float &rhs) { return lhs * std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false); };
    template <class T>
    std::shared_ptr<tensor<T>> operator*=(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs * rhs;
    };
    template <class T>
    std::shared_ptr<tensor<T>> operator*=(std::shared_ptr<tensor<T>> lhs, const float &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs * rhs;
    };

    template <class T>
    std::shared_ptr<tensor<T>> operator-(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> rhs) { return lhs + -rhs; };
    template <class T>
    std::shared_ptr<tensor<T>> operator-(std::shared_ptr<tensor<T>> lhs, const float &rhs)
    {
        auto rhs_tensor = std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false);
        return lhs - rhs_tensor;
    };
    template <class T>
    std::shared_ptr<tensor<T>> operator-=(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs - rhs;
    };
    template <class T>
    std::shared_ptr<tensor<T>> operator-=(std::shared_ptr<tensor<T>> lhs, const float &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs - rhs;
    };

    template <class T>
    std::shared_ptr<tensor<T>> operator/(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> rhs) { return lhs->div(rhs); };
    template <class T>
    std::shared_ptr<tensor<T>> operator/(std::shared_ptr<tensor<T>> lhs, const float &rhs)
    {
        auto rhs_tensor = std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false);
        return lhs / rhs_tensor;
    };
    template <class T>
    std::shared_ptr<tensor<T>> operator/=(std::shared_ptr<tensor<T>> lhs, const float &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs / rhs;
    };
    template <class T>
    std::shared_ptr<tensor<T>> operator/=(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> &rhs)
    {
        CHECK_ARGS_IN_PLACE_OPS(lhs);
        return lhs / rhs;
    };
}
#endif