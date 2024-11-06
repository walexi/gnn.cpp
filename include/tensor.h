#ifndef TENSOR_H
#define TENSOR_H

#include "utils.h"
#include "operation.h"
#include "valarray"
#include "array"
#include <new>
#include <iostream>
#include <memory>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <sstream>
#include <type_traits>
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp));

namespace cyg
{

    enum class Device : short
    {
        cpu,
        cuda
    };

    template <class T>
    class tensor : public std::enable_shared_from_this<tensor<T>>
    {
    private:
        std::valarray<T> *d; // good enough to use just valarray, wo unique_ptr
        std::vector<size_t> dims;
        std::valarray<float> *grad;
        Device device;
        bool requires_grad;

    public:
        typedef T value_type;
        std::shared_ptr<Operation<tensor<T>>> grad_fn;
        explicit tensor(std::vector<size_t> dims, Device device = Device::cpu, bool requires_grad = false) : dims(dims), device(device), requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            this->d = initialize<T>(dims, 1);
            if (requires_grad)
            {
                if (typeid(T) != typeid(float &))
                    throw std::runtime_error(ERROR_GRAD_DTYPE);
                this->zero_grad();
            }
        };
        explicit tensor(std::vector<size_t> dims, int d=0, Device device = Device::cpu, bool requires_grad = false) : dims(dims), device(device), requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            this->d = initialize<T>(dims, d);
            if (requires_grad)
            {
                if (typeid(T) != typeid(float &))
                    throw std::runtime_error(ERROR_GRAD_DTYPE);
                this->zero_grad();
            }
        };
        explicit tensor(std::valarray<T> &data, std::vector<size_t> dims, Device device = Device::cpu, bool requires_grad = false) : d(&data), dims(dims), device(device), requires_grad(requires_grad)
        {
            CHECK_VALID_DIMS(dims);
            CHECK_SIZE(dims, data.size());

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
                this->grad = nullptr;
            };
        };
        // // tensor(const tensor& other):grad_fn(other.grad_fn), d(other.d), grad(other.grad), device(other.device), requires_grad(other.requires_grad), _prev(other._prev),dims(other.dims){}
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
            this->grad = initialize<float>(this->dims, 0);
        };
        const Device get_device() const { return this->device; }
        /**
         * @brief remove dim with size of 1 from the give tensor
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
            bool isvalid = -this->rank() - 1 <= dim && dim <= this->rank();
            std::string message = std::format(std::string_view("dim must be within range of [-{0}-1, {0}]"), this->rank());
            if (!isvalid)
            {
                assert(message);
                throw std::runtime_error(message);
            }
            this->dims.insert(this->dims.begin() + (dim < 0 ? dim + this->rank() + 1 : dim), 1);
            return this->shared_from_this();
        };
        /**
         * @brief class method to backprop on given tensor
         *
         * @param incoming_gradient(type std::vector<float>)
         */
        void backward(tensor<T>* incoming_gradient = nullptr)
        {
            // assertm(this->requires_grad, "please enable requires_grad on tensor to backprop");
            if (incoming_gradient == nullptr && this->n_elements() != 1)
                throw std::runtime_error(ERROR_NON_SCALAR_BACKPROP);
            // 2 scenarios
            // incoming grad, prolly no mistmatch
            // scalar
            if (incoming_gradient->n_elements() != this->n_elements())
                throw std::runtime_error(ERROR_GRAD_MISMATCH);
            if (incoming_gradient == nullptr)
                incoming_gradient = std::make_shared<tensor<T>>(this->dims, 1).get(); // scalar, size=1
            if (this->grad != nullptr)
                *this->grad += *incoming_gradient->data();

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
            CHECK_RANK(other->shape(), this->shape());
            CHECK_SIZE(other->shape(), this->n_elements());
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
            CHECK_RANK(other->shape(), this->shape());
            CHECK_SIZE(other->shape(), this->n_elements());
            auto mul_op = std::make_shared<Mul<tensor<T>>>();
            auto output = mul_op->forward(this->shared_from_this(), other);
            if (this->requires_grad)
                output->grad_fn = mul_op;
            return output;
        };
        std::shared_ptr<tensor<T>> mm(const std::shared_ptr<tensor<T>> other){
            CHECK_MM_DIMS(this->shape(), other->shape());
            // CHECK_NO_BROADCAST(this->shape(), other->shape());
            auto mat_mul_op = std::make_shared<MatMul<tensor<T>>>();
            auto output = mat_mul_op->forward(this->shared_from_this(), other);
            if(this->requires_grad) output->grad_fn = mat_mul_op;

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
            CHECK_RANK(other->shape(), this->shape());
            CHECK_SIZE(other->shape(), this->n_elements());
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
        std::shared_ptr<tensor<T>> pow(const std::shared_ptr<tensor<T>> exponent)
        {
            CHECK_RANK(exponent->shape(), this->shape());
            CHECK_SIZE(exponent->shape(), this->n_elements());
            auto pow_op =std::make_shared<Pow<tensor<T>>>();
            auto output = pow_op->forward(this->shared_from_this(), exponent);
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
        std::shared_ptr<tensor<T>> pow(const float &exponent) //@todo use template
        {
            const auto t = std::make_shared<tensor<T>>(this->dims, exponent, this->device, this->requires_grad);
            auto out = this->pow(t);
            return out;
        }
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
        std::shared_ptr<tensor<T>> mean(const int &dim=-1, const bool &keepdims = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -1);
            auto mean_op = std::make_shared<Mean<tensor<T>>>();
            auto output = mean_op->forward(this->shared_from_this(), dims, keepdims);
            if (this->requires_grad)
                output->grad_fn = mean_op;
            return output;
        };
        /**
         * @brief compute the exponent of a tensor along
         * for ex
         *
         *
         * @return tensor (type std::shared_ptr<cyg::tensor>)
         */
        std::shared_ptr<tensor<T>> exp()
        {
            auto exp_op = std::make_shared<Exp<tensor<T>>>();
            auto output = exp_op->forward(this->shared_from_this());
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
        std::shared_ptr<tensor<T>> log()
        {
            auto log_op = std::make_shared<Log<tensor<T>>>();
            auto output = log_op->forward(this->shared_from_this());
            if (this->requires_grad)
                output->grad_fn = log_op;
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
        std::shared_ptr<tensor<T>> sum(const int &dim = -1, const bool &keepdim = false)
        {
            CHECK_VALID_RANGE(dim, this->rank(), -1);
            auto sum_op = std::make_shared<Sum<tensor<T>>>();
            auto output = sum_op->forward(this->shared_from_this(), dim, keepdim);
            if (this->requires_grad)
                output->grad_fn = sum_op;
            return output;
        };
        std::shared_ptr<tensor<int>> argmax(const int &dim = -1, const bool &keepdim = false)
        {
            std::vector<size_t> new_dims;
            std::valarray<int> id_data;
            if (dim == -1)
                id_data.resize(1);
            else
                id_data.resize(this->dims[dim]);

            std::iota(std::begin(id_data), std::end(id_data), 0);

            auto data = (*this->d);

            auto out_data = new (std::nothrow) std::valarray<int>(1);
            if (out_data == nullptr)
                throw std::runtime_error("insufficient memory");

            if (dim == -1)
            {
                *out_data = id_data[data == data.max()];
                new_dims = {1};
            }
            else
            {

                std::valarray<size_t> strides, sizes, start_idxs;
                std::tie(strides, sizes, start_idxs) = generate_idxs(this->shape(), dim);

                out_data->resize(start_idxs.size());
                for (auto i = 0; const auto &idx : start_idxs)
                {
                    auto gslice = std::slice(idx, this->shape()[dim], strides[dim]);
                    auto data_slice = std::valarray(data[gslice]);
                    (*out_data)[i++] = std::valarray(id_data[data_slice == data_slice.max()])[0];
                }
                new_dims.assign(std::begin(sizes), std::end(sizes));
            }

            auto output = std::make_shared<tensor<int>>(*out_data, new_dims, this->device, false);
            if (!keepdim && dim != -1)
                output->squeeze();
            return output;
        };
        void triu()//inplace op
        {
            // number of elements below main diagonal ==
            std::valarray<size_t> idxs;
            std::tie(std::ignore, std::ignore, idxs) = generate_idxs(this->shape(), this->rank() - 1);
            int itr = 1;
            for (int i = 1; i < idxs.size(); ++i)
            {
                itr %= this->shape()[this->rank() - 2];
                auto gsl = std::slice(idxs[i], itr, 1);
                (*this->d)[gsl] = 0;
                if(this->requires_grad) (*this->grad)[gsl] = 0;
                itr++;
            }
        }; 
        // main diagonal for now i.e diagonal=0;
        std::shared_ptr<tensor<T>> transpose(int a = 0, int b = 1){
            //col with row
            //cant transponse 1D = asert dims>=2
            CHECK_TRANSPOSE(this->shape(), a, b);
            auto tr_op = std::make_shared<Transpose<tensor<T>>>();
            auto output = tr_op->forward(this->shared_from_this(), a, b);
            if(this->requires_grad) output->grad_fn = tr_op;
            return output;
        };

        std::shared_ptr<tensor<T>> var(const int& dim=-1, const bool& keepdim=true){
            CHECK_VALID_RANGE(dim, this->rank(), -1);
            auto var_op = std::make_shared<Var<tensor<T>>>();
            auto output = var_op->forward(this->shared_from_this(), dim, keepdim);
            if(this->requires_grad) output->grad_fn = var_op;
            return output;
        }

        void uniform(const float& low, const float& high){ // inplace
            std::generate(begin((*this->d)), end((*this->d)), [low, high](){ return generate_random(low, high); });
        }
        void repeat(const int& dim, const int& n_repeat){//in place op
            CHECK_VALID_RANGE(dim, this->rank());
            this->dims[dim]=n_repeat;
            int n_elements = std::accumulate(this->dims.begin(), this->dims.end(), 1, std::multiplies<int>());
            auto out_data = new (std::nothrow) std::valarray<T>(n_elements);
            auto grad_data = new (std::nothrow) std::valarray<float>(n_elements);
            if(grad_data==nullptr || out_data==nullptr) throw std::runtime_error("insufficient memory");
            std::valarray<size_t> strides, idxs;
            std::tie(strides, idxs) = generate_idxs(this->dims, dim);

            for(int i=0; const auto& id:idxs)
            {
                (*out_data)[std::slice(id, n_repeat, strides[dim])] = (*this->d)[i];
                (*grad_data)[std::slice(id, n_repeat, strides[dim])] = (*this->grad)[i++];
                // (*out_d)[std::slice(id, )]
            }
            this->d = out_data;
            this->grad = grad_data;
        }
        
        // tensor(tensor&& other);
        // tensor& operator=(tensor&& other);
        // disable copy constructor and copy assigment operator
        // tensor(const tensor&);
        // tensor& operator=(const tensor&);
    };
    
    
    
    /**
     * @brief create a tensor with randomly generated data from a uniform distribution
     * @ todo add a param for a generator (different distributions)
     *
     * @param dims(type std::vector<int>)
     * @param low(type int)
     * @param high(type int)
     * @param device(type cyg::Device)
     * @param requires_grad(type bool)
     *
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    std::shared_ptr<tensor<float>> randn(std::vector<size_t> &dims, int low = -1, int high = 1, Device device = Device::cpu, bool requires_grad = false)
    {
        assertm(low < high, "low must be lower than high, pls check your input params");
        if (low >= high)
            throw std::runtime_error("pls check input params, the value for the low arg must be lower than the high args");
        auto vec = initialize<float>(dims, 1);
        std::generate(begin(*vec), end(*vec), [low, high]()
                      { return generate_random(low, high); });
        return std::make_shared<tensor<float>>(*vec, dims, device, requires_grad);
    };

    /**
     * @brief overloading cout operator for tensor
     *
     * @param out(type std::ostream)
     * @param input_tensor(type cyg::tensor)
     *
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
        out << "), requires_grad = " << std::boolalpha << t.require_grad();
        std::string device = (t.get_device() == Device::cpu) ? "cpu" : "cuda";
        out << ", device = " << device << " )" << "\n";
        return out;
    };

    /**
     * @brief create a tensor with 1s and same shape as input tensor
     *
     *
     * @param input_tensor(type std::shared_ptr<tensor>)
     * @param device(type cyg::Device)
     * @param requires_grad(type bool)
     *
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    template <class T>
    std::shared_ptr<tensor<T>> ones_like(const std::shared_ptr<tensor<T>> &input_tensor, Device device = Device::cpu, bool requires_grad = false)
    {
        return std::make_shared<tensor<T>>(input_tensor->shape(), 1, device, requires_grad);
    };
    /**
     * @brief create a tensor with 0s and same shape as input tensor
     *
     *
     * @param input_tensor(type std::shared_ptr<tensor>)
     * @param device(type cyg::Device)
     * @param requires_grad(type bool)
     *
     * @return generated tensor(type std::shared_ptr<tensor>)
     */
    template <class T>
    std::shared_ptr<tensor<T>> zeros_like(const std::shared_ptr<tensor<T>> &input_tensor, Device device = Device::cpu, bool requires_grad = false)
    {
        return std::make_shared<tensor<T>>(input_tensor->shape(), 0, device, requires_grad);
    };

    template <class T>
    std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> lhs, const std::shared_ptr<tensor<T>> rhs) { return lhs->add(rhs); };
    template <class T>
    std::shared_ptr<tensor<T>> operator+(std::shared_ptr<tensor<T>> lhs, const float &rhs) { return lhs + std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), lhs->get_device(), false); };
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
    std::shared_ptr<tensor<T>> operator*(std::shared_ptr<tensor<T>> lhs, const float &rhs) { return lhs * std::make_shared<tensor<T>>(lhs->shape(),  static_cast<T>(rhs), lhs->get_device(), false); };
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
        auto rhs_tensor = std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), lhs->get_device(), false);
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
        auto rhs_tensor = std::make_shared<tensor<T>>(lhs->shape(), static_cast<T>(rhs), lhs->get_device(), false);
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