#ifndef TENSOR_H
#define TENSOR_H

#include "utils.h"
#include "operation.h"
#include "functional.h"
#include <valarray>
#include <iostream>
#include <memory>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <sstream>
#include <type_traits>

// TODO integrate cuda - thrust -  tensor._data and tensor._grad
// TODO use herogenous containers (boost, variant)
// TODO copy/move constructor/assignment
// TODO fix slice backprop and implement stack and cat operations
// TODO trainer class
// TODO add logging
namespace cyg
{

    template <class T>
    class tensor;

    template <class A>
    using tptr = std::shared_ptr<tensor<A>>;

    template <class T>
    class tensor : public std::enable_shared_from_this<tensor<T>>
    {

        template <class A>
        friend tptr<T> operator+(tptr<T> lhs, const A &rhs) { return lhs->add(rhs); };
        template <class A>
        friend tptr<T> operator+=(tptr<T> lhs, const A &rhs)
        {
            CHECK_ARGS_IN_PLACE_OPS(*lhs);
            auto out = lhs + rhs;
            lhs->set_data(out->data());
            return lhs;
        };

        friend tptr<T> operator-(const tptr<T> &lhs) { return lhs->mul(-1); };
        template <class A>
        friend tptr<T> operator-(tptr<T> lhs, const A &rhs) { return lhs->add(-rhs); };
        template <class A>
        friend tptr<T> operator-=(tptr<T> lhs, const A &rhs)
        {
            CHECK_ARGS_IN_PLACE_OPS(*lhs);
            auto out = lhs - rhs;
            lhs->set_data(out->data());
            return lhs;
        };
        friend tptr<T> operator*(tptr<T> lhs, const tptr<T> &rhs) { return lhs->mul(rhs); };
        friend tptr<T> operator*(tptr<T> lhs, const float &rhs) { return lhs->mul(rhs); };
        friend tptr<T> operator*=(tptr<T> lhs, const tptr<T> &rhs)
        {
            CHECK_ARGS_IN_PLACE_OPS(*lhs);
            auto out = lhs * rhs;
            lhs->set_data(out->data());
            return lhs;
        };
        friend tptr<T> operator*=(tptr<T> lhs, const float &rhs)
        {
            CHECK_ARGS_IN_PLACE_OPS(*lhs);
            return lhs *= (std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false));
        };

        friend tptr<T> operator/(tptr<T> lhs, const tptr<T> rhs) { return lhs->div(rhs); };
        friend tptr<T> operator/(tptr<T> lhs, const float &rhs) { return lhs->div(rhs); }
        friend tptr<T> operator/=(tptr<T> lhs, const tptr<T> &rhs)
        {
            CHECK_ARGS_IN_PLACE_OPS(*lhs);
            auto out = lhs / rhs;
            lhs->set_data(out->data());
            return lhs;
        };
        friend tptr<T> operator/=(tptr<T> lhs, const float &rhs)
        {
            CHECK_ARGS_IN_PLACE_OPS(*lhs);
            return lhs /= (std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false));
        };
        friend tptr<bool> operator>(tptr<T> lhs, const tptr<T> &rhs)
        {
            return lhs->gt(rhs);
        };
        friend tptr<bool> operator>(tptr<T> lhs, const float &rhs)
        {
            return lhs > rhs;
        };
        friend tptr<bool> operator<(tptr<T> lhs, const tptr<T> &rhs)
        {
            return rhs > lhs;
        };
        friend tptr<bool> operator<(tptr<T> lhs, const float &rhs)
        {
            auto rhs_tensor = std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), false);
            return rhs_tensor->gt(lhs);
        };

    public:
        typedef T value_type;
        std::unique_ptr<cyg::Operation<tensor<T>>> grad_fn;

        /**
         * @brief constructor for the tensor class
         *
         * @param dims(type std::vector<size_t>)
         * @param value(type T)
         * @param requires_grad(type bool)
         */
        explicit tensor(std::vector<size_t> dims, T value = 0, bool requires_grad = false) : _data(initialize<T>(dims, value)), _dims(dims), _requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            if (requires_grad)
            {
                if (typeid(T) != typeid(float &))
                    throw std::runtime_error(ERROR_GRAD_DTYPE);
                this->zero_grad();
            }
        };

        /**
         * @brief constructor for the tensor class
         *
         * @param dims(type std::vector<size_t>)
         * @param data(type std::valarray<T>*)
         * @param requires_grad(type bool)
         */
        explicit tensor(std::vector<size_t> dims, std::valarray<T> *data = nullptr, bool requires_grad = false) : _dims(dims), _requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            delete this->_data;
            if (data != nullptr)
            {
                CHECK_SIZE(dims, data->size());
                this->_data = data;
            }
            else
                this->_data = initialize<T>(dims, 0);
            if (requires_grad)
            {
                if (typeid(T) != typeid(float &))
                    throw std::runtime_error(ERROR_GRAD_DTYPE);
                this->zero_grad();
            }
        };
        // copy constructor
        //  tensor(const tensor<T>& other): {
        //      delete _data; delete _grad;
        //      _data = other.data();
        //      _requires_grad = other.requires_grad();
        //      _dims = other.shape();
        //      //disable copy for tensor with req grad true, dont think it make sense
        //  };
        //  //copy assignment
        //  tensor& operator=(const tensor<T> other){
        //      if(this != &other){
        //          delete _data, _grad;
        //          _data = other.data();
        //          _requires_grad = other.requires_grad();
        //          _dims = other.shape();
        //      }
        //      //disable copy for tensor with req grad true, dont think it make sense
        //      return *this;
        //  };
        //  ~tensor() noexcept { delete _data; delete _grad; grad_fn->reset(); }; // destructor must not fail - so noexcept
        std::valarray<T> *data() const { return this->_data; };
        template <class A>
        void set_data(std::valarray<A> *data)
        {
            if (data->size() != this->numel())
                throw std::runtime_error(ERROR_SIZE_MISMATCH);

            for (int i = 0; i < this->numel(); i++)
                (*this->_data)[i] = static_cast<T>((*data)[i]);

            if (this->_requires_grad)
                this->zero_grad();
        };
        template <class A>
        operator tensor<A>() const
        {
            auto new_data = new std::valarray<A>(this->numel());
            for (int i = 0; i < this->numel(); i++)
                (*new_data)[i] = static_cast<A>((*this->_data)[i]);
            return tensor<A>(this->_dims, new_data, this->_requires_grad);
        };
        std::shared_ptr<tensor<float>> to_float()
        { // inplace for casting = tensors wo grad
            // if (typeid(T) == typeid(float &))
            //     return this->shared_from_this();
            return std::make_shared<tensor<float>>((tensor<float>)(*this));
        }
        std::shared_ptr<tensor<int>> to_int()
        {
            return std::make_shared<tensor<int>>((tensor<int>)(*this));
        }
        std::shared_ptr<tensor<bool>> to_bool()
        {
            if (typeid(T) == typeid(bool &))
                return this->shared_from_this();
            return std::make_shared<tensor<bool>>((tensor<bool>)(*this));
        }
        std::shared_ptr<tensor<double>> to_double()
        {
            if (typeid(T) == typeid(double &))
                return this->shared_from_this();
            return std::make_shared<tensor<double>>((tensor<double>)(*this));
        }
        std::vector<size_t> shape() const { return this->_dims; };
        const size_t numel() const { return this->_data->size(); };
        // const int count_nonzero() const { return (this->_data == 0).sum();}
        const int rank() const { return this->_dims.size(); };

        /**
         * @brief return grad of a leaf tensor
         */
        std::valarray<float> *grad()
        {
            if (grad_fn != nullptr)
                throw std::runtime_error(WARNING_GRAD_NOT_LEAF);
            if (_grad == nullptr)
                throw std::runtime_error("invalid op, pls enable grad on this tensor");
            return this->_grad;
        };
        const bool requires_grad() const { return this->_requires_grad; };
        /**
         * @brief class method to turn on/off _grad for the given tensor
         *
         * @param requires_grad(type bool)
         */
        void requires_grad_(bool requires_grad)
        {
            if (requires_grad)
            {
                this->_requires_grad = true;
                this->zero_grad();
            }
            else
            {
                this->_requires_grad = false;
                delete this->_grad;
                this->_grad = nullptr;
            };
        };
        void zero_grad()
        {
            delete this->_grad;
            this->_grad = initialize<float>(this->_dims, 0);
        };
        /**
         * @brief remove dim with size of 1 from the given tensor
         *
         */
        void squeeze()
        {
            this->_dims.erase(std::remove(this->_dims.begin(), this->_dims.end(), 1), this->_dims.end());
        };
        /**
         * @brief add dim of size 1 at the input dim of the given tensor
         *
         * @param dim(type int)
         */
        tptr<T> unsqueeze(int dim)
        {
            CHECK_VALID_RANGE(dim, this->rank() + 1, -this->rank() - 1);
            this->_dims.insert(this->_dims.begin() + (dim < 0 ? dim + this->rank() + 1 : dim), 1);
            return this->shared_from_this();
        };
        /**
         * @brief class method to backprop on given tensor
         *
         * @param incoming_gradient(type std::vector<float>)
         */
        void backward(std::shared_ptr<tensor<float>> incoming_gradient = nullptr)
        {
            // assertm(this->_requires_grad, "please enable requires_grad on tensor to backprop");
            if (incoming_gradient == nullptr && this->numel() != 1)
                throw std::runtime_error(ERROR_NON_SCALAR_BACKPROP);

            if (incoming_gradient == nullptr)
                incoming_gradient = make_shared<tensor<float>>(this->_dims, 1, false); // scalar, size=1

            if (incoming_gradient->numel() != this->numel())
                throw std::runtime_error(ERROR_GRAD_MISMATCH);

            if (this->_grad != nullptr)
            {
                *this->_grad += *incoming_gradient->data();
            }
            if (this->grad_fn != nullptr)
            {
                // if(this->grad_fn->_done==false && this->numel()==1){
                //     std::cout<<"trying to backprop on this node again, pls be sure this is intended"<<"\n";
                // }
                this->grad_fn->backward(incoming_gradient);
            }
        };
        /**
         * @brief operator overloading for the function call to index a tensor
         *
         * @return value at the input index (type T)
         */
        T operator()(size_t d, ...) const
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
            CHECK_VALID_INDEX(dims, _dims);
            auto index = get_index(_dims, dims);
            return (*this->_data)[index];
        };
        auto &operator[](size_t dim) const // index a 1D tensor
        {
            if (this->rank() != 1)
                throw std::runtime_error("tensor must be 1D");
            if (dim >= this->numel())
                throw std::runtime_error("invalid index");
            return (*_data)[dim];
        }

        /**
         * @brief addition operation for tensors, tensors must have equal shape or be broadcastable
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        tptr<T> add(const tptr<T> &other)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), other->shape());
            auto add_op = std::make_unique<cyg::Add<tensor<T>>>();
            auto output = add_op->forward(this->shared_from_this(), other);
            if (output->requires_grad())
                output->grad_fn = std::move(add_op);
            return output;
        };
        /**
         * @brief addition operation for tensors, tensors must have equal shape or be broadcastable
         *
         * @param other(type: A)
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        template <class A>
        tptr<T> add(const A &other)
        {
            auto other_tensor = std::make_shared<tensor<T>>(this->_dims, static_cast<T>(other), false);
            return this->add(other_tensor);
        }
        /**
         * @brief element wise multiplication operation for tensors, tensors must have equal shape or be broadcastable
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        tptr<T> mul(const tptr<T> &other)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), other->shape());
            auto mul_op = std::make_unique<cyg::Mul<tensor<T>>>();
            auto output = mul_op->forward(this->shared_from_this(), other);
            if (output->requires_grad())
                output->grad_fn = std::move(mul_op);
            return output;
        };
        /**
         * @brief element wise multiplication operation for tensors, tensors must have equal shape or be broadcastable
         *
         * @param other(type: A)
         * @return tensor (type cyg::tensor)
         */
        template <class A>
        tptr<T> mul(const A &other)
        {
            auto other_tensor = std::make_shared<tensor<T>>(this->_dims, static_cast<T>(other), false);
            return this->mul(other_tensor);
        };

        tptr<T> mm(const tptr<T> &other)
        {
            CHECK_MM_DIMS(this->shape(), other->shape());
            auto mat_mul_op = std::make_unique<cyg::MatMul<tensor<T>>>();
            auto output = mat_mul_op->forward(this->shared_from_this(), other);
            if (output->requires_grad())
                output->grad_fn = std::move(mat_mul_op);

            return output;
        };

        /**
         * @brief element wise division operation for tensors, tensors must have equal shape
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        tptr<T> div(const tptr<T> &other)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), other->shape());
            auto div_op = std::make_unique<cyg::Div<tensor<T>>>();
            auto output = div_op->forward(this->shared_from_this(), other);
            if (output->requires_grad())
                output->grad_fn = std::move(div_op);

            return output;
        };
        /**
         * @brief element wise division operation for tensors, tensors must have equal shape
         *
         * @param other(type: A)
         * @return tensor (type cyg::tensor)
         */
        template <class A>
        tptr<T> div(const A &denominator)
        {
            const auto denominator_tensor = std::make_shared<tensor<T>>(this->_dims, static_cast<T>(denominator), false); //@todo use named args for constructor
            return this->div(denominator_tensor);
        }
        /**
         * @brief element wise exponent operation for tensors, tensors must have equal shape
         *
         * @param other(type cyg::tensor)
         * @param inplace(type bool)
         * @return tensor (type cyg::tensor<T>)
         */
        tptr<T> pow(const tptr<T> &exponent, const bool &inplace = false)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), exponent->shape());
            auto pow_op = std::make_unique<cyg::Pow<tensor<T>>>();
            auto output = pow_op->forward(this->shared_from_this(), exponent);
            if (output->requires_grad())
                output->grad_fn = std::move(pow_op);
            if (inplace)
            {
                CHECK_ARGS_IN_PLACE_OPS(*this);
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }

            return output;
        };

        /**
         * @brief scalar exponent operation for tensors, raise a tensor to a scalar power
         *
         * @param other(type: cyg::tensor)
         * @return tensor (type cyg::tensor)
         */
        template <class A>
        tptr<T> pow(const A &exponent)
        {
            const auto exponent_tensor = std::make_shared<tensor<T>>(this->_dims, static_cast<T>(exponent), false); //@todo use named args for constructor
            return this->pow(exponent_tensor);
        };

        /**
         * @brief yield input where condition is true (non zero) otherwise yield other
         * @param bool_tensor(type std::shared_ptr<cyg::tensor<bool>>>)
         * @param other(type std::shared_ptr<cyg::tensor<T>>>)
         */
        tptr<T> where(const tptr<bool> bool_tensor, const tptr<T> &other)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), bool_tensor->shape());
            CHECK_ARGS_OPS_BROADCAST(this->shape(), other->shape());
            auto mask_op = std::make_unique<Mask<tensor<T>>>();
            auto output = mask_op->forward(bool_tensor->to_float(), this->shared_from_this(), other);
            if (output->requires_grad())
                output->grad_fn = std::move(mask_op);
            return output;
        }
        /**
         * @brief yield input where condition is true (non zero) otherwise yield other
         * @param bool_tensor(type std::shared_ptr<cyg::tensor<bool>>>)
         * @param other(type std::shared_ptr<cyg::tensor<T>>>)
         */
        template <class A>
        tptr<T> where(const tptr<bool> &bool_tensor, const A &other)
        {
            auto other_tensor = make_shared<tensor<T>>(this->_dims, static_cast<T>(other), false);
            return this->where(bool_tensor, other_tensor);
        };

        /**
         * @brief compute the exponent of a tensor along
         * for ex
         * @param inplace(type bool)
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        tptr<T> exp(const bool &inplace = false)
        {
            auto exp_op = std::make_unique<cyg::Exp<tensor<T>>>();
            auto output = exp_op->forward(this->shared_from_this());
            if (output->requires_grad())
                output->grad_fn = std::move(exp_op);
            if (inplace)
            {
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }
            return output;
        };

        /**
         * @brief compute the natural log of a tensor along
         * for ex
         * @param inplace(type bool)
         * @return tensor (type std::shared_ptr<cyg::tensor<T>>)
         */
        tptr<T> log(const bool &inplace = false)
        {
            auto log_op = std::make_unique<Log<tensor<T>>>();
            auto output = log_op->forward(this->shared_from_this());
            if (inplace)
            {
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }
            if (output->requires_grad())
                output->grad_fn = std::move(log_op);
            return output;
        };

        /**
         * @brief compute the mean along the input dimension if specified, otherwise over all the elements in the tensor
         * for ex
         *  auto t = cyg::ones({2,3,4})
         *  t.mean(2) // mean along the dimension of size 4
         *
         * @param dim(type: int)
         * @param keepdims(type: bool)
         * @param inplace(type: bool)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<T>>)
         */
        tptr<float> mean(int dim = INT_MAX, const bool &keepdim = false, const bool &inplace = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            auto mean_op = std::make_unique<Mean<tensor<float>>>();
            auto output = mean_op->forward(this->shared_from_this(), dim, keepdim);
            if (inplace)
            {
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }
            if (output->requires_grad())
                output->grad_fn = std::move(mean_op);

            return output;
        };

        /**
         * @brief compute the sum of each row of the given tensor in the input dimension if specified, otherwise over all the elements in the tensor
         * for ex
         *
         * @param dim(type int)
         * @param keepdim(type bool)
         * @param inplace(type: bool)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        tptr<T> sum(int dim = INT_MAX, const bool &keepdim = false, const bool &inplace = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());

            auto sum_op = std::make_unique<cyg::Sum<tensor<T>>>();
            auto output = sum_op->forward(this->shared_from_this(), dim, keepdim);
            if (inplace)
            {
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }
            if (output->requires_grad())
                output->grad_fn = std::move(sum_op);

            return output;
        };

        /**
         * @brief transpose tensor
         *
         * @param d1(type int)
         * @param d2(type int)
         * @param inplace(type: bool)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<T>>)
         */
        tptr<T> t(int d1 = -1, int d2 = -2, const bool &inplace = false)
        {
            CHECK_TRANSPOSE(this->shape(), d1, d2);
            auto tr_op = std::make_unique<Transpose<tensor<T>>>();
            auto output = tr_op->forward(this->shared_from_this(), d1, d2);
            if (inplace)
            {
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }
            if (output->requires_grad())
                output->grad_fn = std::move(tr_op);

            return output;
        };
        /**
         * @brief compute variance of tensor along a dim if specified, otherwise over all the elements in the tensor
         *
         * @param dim(type int)
         * @param d2(type int)
         * @param inplace(type: bool)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<T>>)
         */
        tptr<float> var(int dim = INT_MAX, const int &correction = 1, const bool &keepdim = true, const bool &inplace = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());

            auto var_op = std::make_unique<Var<tensor<float>>>();
            auto output = var_op->forward(this->shared_from_this(), dim, correction, keepdim);
            if (inplace)
            {
                delete this->_data;
                this->_data = output->data();
                this->_dims = output->shape();
                return this->shared_from_this();
            }
            if (output->requires_grad())
                output->grad_fn = std::move(var_op);

            return output;
        }

        /**
         * @brief sum tensor to size
         *
         * @param dims(type istd::vector<size_t>)
         *
         */
        void sum_to_size(std::vector<size_t> dims)
        { // inplace op, dims must be broadcastable to this.shape, this.shape>=dims
            CHECK_ARGS_IN_PLACE_OPS(*this);
            if (this->rank() < dims.size())
                throw std::runtime_error("not expandable");
            if (!is_broadcastable(this->_dims, dims))
                throw std::runtime_error("dims is not broacastable to this tensor's size");
            int n_iterations = -this->rank();
            for (int i = -1; i >= n_iterations; i--)
            {
                if (std::abs(i) > dims.size())
                    this->sum(0, false, true);
                else
                {
                    if (this->_dims[this->rank() + i] < dims[dims.size() + i])
                        throw std::runtime_error(" not expandable");
                    if (dims[dims.size() + i] != this->_dims[this->rank() + i])
                        this->sum(i, true, true);
                }
            }
        }
        /**
         * @brief argmax of tensor along dim if specified otherwise over all the elements
         *
         * @param dim(type int)
         * @param keepdim(type bool)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<int>>)
         */
        tptr<int> argmax(int dim = INT_MAX, const bool &keepdim = false)
        {
            return std::get<1>(this->max(dim, keepdim));
        };

        /**
         * @brief return element in scalar tensor
         * @return value (type T)
         */
        T item()
        {
            if (this->numel() != 1)
                throw std::runtime_error("invalid op, tensor must be scalar");
            return (*_data)[0];
        };
        /**
         * @brief max of tensor along dim if specified otherwise over all the elements
         *
         * @param dim(type int)
         * @param keepdim(type bool)
         *
         * @return tuple of max values and the indices (type std::tuple<std::shared_ptr<tensor<T>>, std::shared_ptr<tensor<int>>>)
         */
        std::tuple<tptr<T>, tptr<int>> max(int dim = INT_MAX, const bool &keepdim = false) const
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            // TODO move to operation to compute backprop for max_values
            auto [max_values, indices] = functional::max(*this, dim, keepdim);

            return {max_values, indices};
        };

        /**
         * @brief set elements of a 2D tensor below the main diagonal to zero
         */
        void triu() // inplace op
        {
            CHECK_ARGS_IN_PLACE_OPS(*this);
            // number of elements below main diagonal ==
            const auto [_, idxs] = generate_idxs(this->shape(), -1);
            int itr = 1;
            for (int i = 1; i < idxs.size(); ++i)
            {
                itr %= this->shape()[this->rank() - 2];
                auto gsl = std::slice(idxs[i], itr, 1);
                (*this->_data)[gsl] = 0;
                itr++;
            }
        };
        /**
         * @brief initialize tensor's values from a uniform distribution
         *
         * @param low(type float)
         * @param high(type float)
         */
        void uniform(const float &low, const float &high)
        { // inplace
            std::generate(std::begin((*this->_data)), std::end((*this->_data)), [low, high]()
                          { return generate_random(low, high); });
        }
        /**
         * @brief initialize tensor's values from a uniform distribution
         *
         * @param low(type float)
         * @param high(type float)
         */
        void repeat(int dim, const size_t &n_repeat)
        { // in place op
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            dim < 0 ? dim = this->rank() + dim : dim;
            this->_dims[dim] = n_repeat;

            auto new_d = repeat_nd<T>(this->_data, this->_dims, {{size_t(dim), n_repeat}});
            delete this->_data;
            this->_data = new_d;
            if (this->_requires_grad)
            {
                auto new_grad = repeat_nd<float>(this->_grad, this->_dims, {{size_t(dim), n_repeat}});
                delete this->_grad;
                this->_grad = new_grad;
            }
        }
        /**
         * @brief t1>t2
         * true where elements of t1 is greater than elements of t2 else false
         */
        std::shared_ptr<tensor<bool>> gt(const tptr<T> &other)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), other->shape());
            return functional::gt(*this, *other);
        }

        /**
         * @brief >t2
         * true where elements of t1 is greater than elements of t2 else false
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<bool>>)
         */
        std::shared_ptr<tensor<bool>> gt(const float &other)
        {
            auto other_tensor = tensor<T>(this->_dims, static_cast<T>(other), false);
            return functional::gt(*this, &other_tensor);
        }
        /**
         * @brief t1<t2
         * true where elements of t1 is less than elements of t2 else false
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<bool>>)
         */
        std::shared_ptr<tensor<bool>> lt(const tptr<T> &other)
        {
            CHECK_ARGS_OPS_BROADCAST(this->shape(), other->shape());
            return functional::gt(*other, &this);
        }

        /**
         * @brief initialize tensor's values from a uniform distribution
         *
         * @param require_grad(type bool)
         * @param fillValue(type T)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<T>>)
         */
        tptr<T> clone(const bool &require_grad = false, const T fillValue = INT_MAX) const
        {
            auto data = new std::valarray<T>();
            if (fillValue != INT_MAX)
                *data = fillValue;
            else
                *data = *this->_data;
            return std::make_shared<tensor<T>>(this->_dims, data, require_grad);
        }

        /**
         * @brief index tensor given the 1D tensor
         *
         * @param idx_tensor(type bool)
         * @param dim(type int)
         *
         * @return tensor (type std::shared_ptr<cyg::tensor<T>>)
         */
        tptr<T> at(const tptr<int> &idx_tensor, int dim = -1)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -this->rank());
            auto c = std::find(_dims.rbegin() + 1, _dims.rend(), idx_tensor->numel());
            if (c == _dims.rend())
                throw std::runtime_error("invalid op, input tensor's shape must be compatible with this tensor");
            auto sl_op = std::make_unique<Slice<tensor<T>>>();
            auto output = sl_op->forward(this->shared_from_this(), idx_tensor->to_float(), dim);
            if (output->requires_grad())
                output->grad_fn = std::move(sl_op);
            return output;
        }

        void expand(const std::vector<size_t> &dims)
        {
            if (dims.size() < this->rank())
                throw std::runtime_error("invalid op - tensor cannot be expand to this shape");
            for (int j = -1; j >= -dims.size(); j--)
            {
                if (std::abs(j) > this->rank())
                    this->unsqueeze(0);
                if (this->_dims[this->rank() + j] > dims[dims.size() + j])
                    throw std::runtime_error(" not expandable");
                if (dims[dims.size() + j] == this->_dims[this->rank() + j])
                    continue;

                this->repeat(this->rank() + j, dims[dims.size() + j]);
            }
        }

        /**
         * @brief inplace op to fill tensor's diagonal with give value if specified otherwise 0
         *
         * @param value(type T)
         *
         * */
        template <class A>
        void fill_diagonal_(const A &value = 0) const
        {
            // inplace op
            CHECK_ARGS_IN_PLACE_OPS(*this);
            // make sense for only 2D matrix
            if (this->rank() != 2 || _dims[0] != _dims[1])
                throw std::runtime_error("all dimensions must be of same length and tensor must be 2D");
            int num_elem = _dims[0];
            int stride = std::accumulate(_dims.begin() + 1, _dims.end(), 1, std::multiplies<int>());
            (*_data)[std::slice(0, num_elem, stride + 1)] = static_cast<T>(value);
        }

        void flatten()
        {
            _dims = {1};
        }

    protected:
        std::valarray<T> *_data = nullptr;
        std::vector<size_t> _dims;
        std::valarray<float> *_grad = nullptr;
        bool _requires_grad;
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
        out << "(";
        std::stringstream output_string = printND<T>(t.data(), t.shape());
        out << output_string.str() << ", size = ( ";
        for (const auto &r : t.shape())
            out << r << " ";
        out << "), requires_grad = " << std::boolalpha << t.requires_grad();
        if (t.grad_fn)
            out << ", grad_fn=<" << *t.grad_fn << ">";
        out << " )" << "\n";
        return out;
    };

    /**
     * @brief create a tensor with randomly generated data from a uniform distribution
     * @ todo add a param for a generator (different distributions)
     *
     * @param _dims(type std::vector<int>)
     * @param low(type int)
     * @param high(type int)
     * @param requires_grad(type bool)
     *
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    tptr<float> randn(std::vector<size_t> dims, int low = -1, int high = 1, bool requires_grad = false);

    /**
     * @brief create a tensor with 1s and same shape as input tensor
     *
     *
     * @param input_tensor(type std::shared_ptr<tensor>)
     * @param requires_grad(type bool)
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    template <class T>
    tptr<int> ones_like(const tptr<T> &input_tensor, bool requires_grad = false) { return std::make_shared<tensor<int>>(input_tensor->shape(), 1, requires_grad); };

    /**
     * @brief create a tensor with 0s and same shape as input tensor
     *
     *
     * @param input_tensor(type std::shared_ptr<tensor>)
     * @param requires_grad(type bool)
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    template <class T>
    tptr<int> zeros_like(const tptr<T> &input_tensor, bool requires_grad = false) { return std::make_shared<tensor<int>>(input_tensor->shape(), 0, requires_grad); };

    /**
     * @brief disbale grad for input tensors, similar to 'with torch.no_grad()';
     *  no_grad(ts) => disbale grad
     */
    void no_grad(std::vector<tptr<float>> ts);

    /**
     * @brief enable grad for input tensors, similar to 'with torch.no_grad()';
     *  no_grad(ts) => disbale grad
     */
    void enable_grad(std::vector<tptr<float>> ts);
    /**
     * @brief returns a 2D tensor with 1s on the diagonal and 0s elsewhere
     * @param n(type size_t) rows
     * @param m(type size_t) columns
     *
     * @return tensor with 1s on the diagonal and 0s elsewhere (shape n * m) (type std::shared_ptr<T>)
     */
    tptr<int> eye(size_t n, size_t m = INT_MAX);

    // diag(shape)
    // /**
    //  * concatenates the input tensors along a new dimension
    //  */
    // template<class T>
    // tptr<T> stack(const std::vector<tptr<T>> ts, int dim=0){
    //     CHECK_CONCAT(ts);
    //     CHECK_VALID_RANGE(dim, ts[0]->rank() , -ts[0]->rank());
    //     auto stack_op = std::make_unique<Stack<tensor<T>>>();
    //     auto output = stack_op->forward(ts, dim);
    //     if(output->requires_grad()) output->grad_fn = std::move(stack_op);
    //     return output;
    // }

}
#endif