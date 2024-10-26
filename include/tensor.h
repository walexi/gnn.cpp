#ifndef TENSOR_H
#define TENSOR_H
#include "arithmetic.h"
#include "operation.h"
#include <new>
#include <memory>
#include <assert.h>
#include <sstream>

// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp));
 
/**
 * using 1d array with an N_D view/access  d_i where i is the ith dim
 * @TODO
 * use template/generics/concept to handle numeric/integral types - int, float, uint..etc
 * consider pimpl - template spec https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#t84-use-a-non-template-core-implementation-to-provide-an-abi-stable-interface
 * check for type promotion and handle appropriately
 */
namespace cyg
{
    enum class Device : short
    {
        cpu,
        cuda
    };
    class tensor: public std::enable_shared_from_this<tensor>
    {
    private:
        const std::vector<float>* d; //vector vs array, i'm going with vector for now, primarily cos of less code, would prefer std::array but i have to write more code to handle dynamic sizes, copying; unique_ptr comes in handy though
        std::vector<int>& dims; //not const cos we can resize, squeeze, etc
        // also consider static if needed or alt cos of graph - why use raw pointers - 
        std::vector<float>* grad; //volatile, not needed now, prolly later for ipc
        Device device;
        bool requires_grad;
        std::vector<std::shared_ptr<tensor>> _prev;

    public:
        Operation<tensor>* grad_fn;
        explicit tensor(std::vector<int>& dims, Device device = Device::cpu, bool requires_grad=false);
        //const ref are the pref way to pass obj unless the obj can possibly be a nullptr
        //implicit conversion to const for data, not going with pointer since data must not be null
        explicit tensor(const std::vector<float>& data, std::vector<int>& dims, Device device = Device::cpu, bool requires_grad = false);
        void* operator new(std::size_t);
        // tensor(const tensor& other):grad_fn(other.grad_fn), d(other.d), grad(other.grad), device(other.device), requires_grad(other.requires_grad), _prev(other._prev),dims(other.dims){}
        ~tensor() noexcept{ delete d; }; // destructor must not fail - so noexcept
        const std::vector<float>* data() const { return this->d; };
        std::vector<int>& shape() const { return this->dims; };
        const std::size_t n_elements() const { return this->d->size(); };
        const int rank() const { return this->dims.size(); };
        std::vector<float> * get_grad() {
            if(this->grad_fn!=nullptr) throw std::runtime_error("UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward()");
            return this->grad;
        };
        const bool require_grad() const { return this->requires_grad; };
        void update_grad(std::vector<float>* grad);
        void zero_grad();
        void detach();
        // // tensor* clone() const {return new tensor(*this);}
        void enable_grad(bool requires_grad);
        const Device get_device() const { return this->device; }
        void remove_child(const std::shared_ptr<tensor>& child);
        void add_child(std::shared_ptr<tensor> child);
        std::vector<std::shared_ptr<tensor>> get_children() { return this->_prev;}; //for test purpose
        void squeeze();
        tensor& unsqueeze(int dim);
        // void transpose(int dim1=0, int dim2=0);
        void backward(std::vector<float>* incoming_grad);
        float operator()(int dims, ...);
        std::shared_ptr<tensor> triu();// main diagonal for now i.e diagonal=0;
        std::shared_ptr<tensor> operator-(){
            return *this * -1;
        };
        std::shared_ptr<tensor> add(const std::shared_ptr<tensor>& other);
        // std::shared_ptr<tensor> operator+=(const std::shared_ptr<tensor> rhs) { return this->add(rhs); };
        friend std::shared_ptr<tensor> operator+(const std::shared_ptr<tensor>& lhs, const std::shared_ptr<tensor>& rhs) { return lhs->add(rhs);}
        std::shared_ptr<tensor> mul(const std::shared_ptr<tensor>  &other);
        // std::shared_ptr<tensor> operator*=(std::shared_ptr<tensor> &rhs) { return this->mul(rhs); };
        // std::shared_ptr<tensor> mm(const std::shared_ptr<tensor>  &other);
        std::shared_ptr<tensor> operator*(const std::shared_ptr<tensor> &rhs){return this->mul(rhs);}; 
        std::shared_ptr<tensor> operator*(const float &rhs); 
        std::shared_ptr<tensor> div(const std::shared_ptr<tensor> &other);
        // std::shared_ptr<tensor>  operator/=(const std::shared_ptr<tensor> &denominator) { return this->div(denominator); };
        friend std::shared_ptr<tensor> operator/(const std::shared_ptr<tensor>& numerator, const std::shared_ptr<tensor> &denominator){ return numerator->div(denominator);};
        std::shared_ptr<tensor> pow(const std::shared_ptr<tensor> &exponent);
        std::shared_ptr<tensor> pow(const float &exponent); //@todo use template
        std::shared_ptr<tensor> mean(int dim, bool keepdims=true);
        std::shared_ptr<tensor> exp();
        std::shared_ptr<tensor> log();
        
        // tensor(tensor&& other);
        // tensor& operator=(tensor&& other);
        // disable copy constructor and copy assigment operator
        // tensor(const tensor&);
        // tensor& operator=(const tensor&);
        
    };

    // void move_tensor(std::shared_ptr<tensor> tensor, Device device);
    std::vector<float>* initialize(std::vector<int> &dims, int value = 1);
    std::shared_ptr<tensor> randn(std::vector<int> &dims, int low=-1, int high=1, Device device = Device::cpu, bool requires_grad = false);
    std::shared_ptr<tensor> fill(const float& value, std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
    std::shared_ptr<tensor> ones(std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
    std::shared_ptr<tensor> zeros(std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
    std::shared_ptr<tensor> ones_like(const std::shared_ptr<tensor>& a, Device device = Device::cpu, bool requires_grad = false);
    std::shared_ptr<tensor> zeros_like(const std::shared_ptr<tensor>& a, Device device = Device::cpu, bool requires_grad = false);
    std::stringstream printND(std::vector<float> dat, std::vector<int> shape);
    std::ostream &operator<<(std::ostream &out, const tensor &t);
}
#endif