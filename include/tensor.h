#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <assert.h>
/**
 * using 1d array with an N_D view/access  d_i where i is the ith dim
 * @TODO
 * use template/generics/concept to handle numeric/integral types - int, float, uint..etc
 * consider pimpl - template spec https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#t84-use-a-non-template-core-implementation-to-provide-an-abi-stable-interface
 * check for type promotion and handle appropriately
 */
namespace cyg
{

    //     std::vector<float> (*func)(std::vector<float>, std::vector<float>);

    enum class Device : short
    {
        cpu,
        cuda
    };

    class tensor
    {
    private:
        std::vector<float> d;
        const std::vector<int> dims;
        // also consider static if needed or alt cos of graph
        std::vector<float> *grad;
        size_t n_dims;
        Device device;
        bool requires_grad;
        tensor _backward();

    public:
        explicit tensor(std::vector<float> &ms, const std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
        // ~tensor() noexcept; // destructor must not fail - so noexcept
        // tensor(tensor&& other);
        // tensor& operator=(tensor&& other);
        // disable copy constructor and copy assigment operator
        // tensor(const tensor&);
        // tensor& operator=(const tensor&);
        const std::vector<int> &shape() const { return this->dims; };
        const std::size_t n_elements() const { return this->d.size(); };
        const std::size_t &rank() const { return this->n_dims; };
        const std::vector<float> &data() const { return this->d; };
        const char require_grad() const { return this->requires_grad; };
        const Device get_device() const { return this->device; }
        void detach();
        void enable_grad(bool requires_grad);
        // void update_grad(vector<float>* g);
        void transpose(int dim1=0, int dim2=0);
        void zero_grad();
        std::vector<float>* get_grad() {return this->grad;};
        void backward(std::vector<float *> grad);
        const float &operator()(int dims, ...);
        // y = ab dy/db = 1 dy/da = 1
        tensor &add(tensor &other); // not using const for other since i will be update the comp graph of both operands
        tensor &operator+=(tensor &rhs) { return this->add(rhs); };
        friend tensor operator+(tensor lhs, tensor &rhs)
        {
            lhs += rhs;
            return lhs;
        };
        // y = a*b, a.mul(b), a*=b
        tensor &mul(const tensor &other);
        tensor &operator*=(const tensor &rhs) { return this->mul(rhs); };
        friend tensor operator*(tensor lhs, const tensor &rhs)
        {
            lhs *= rhs;
            return lhs;
        }; // remember symmetry in all you do, with bin op
        // y = a - b dy/da = 1 dy/db = -1
        tensor &sub(tensor &other);
        tensor &operator-=(tensor &rhs) { return this->sub(rhs); };
        friend tensor operator-(tensor lhs, tensor &rhs)
        {
            lhs -= rhs;
            return lhs;
        };
        // y = a/b a/=b a.div(b)
        tensor &div(tensor &other);
        tensor &operator/=(tensor &rhs) { return this->div(rhs); };
        friend tensor operator/(tensor lhs, tensor &rhs)
        {
            lhs /= rhs;
            return lhs;
        };
        // y = a**p  dy/da = p*a**(p-1)
        tensor &pow(const tensor &other);
        // a>b a.lt(b)
        tensor &lt(const tensor &other) const;
        friend tensor operator<(const tensor &left, const tensor &right) { return left.lt(right); };
    };

    void move_tensor(tensor &tensor, Device device);
    const std::vector<float> &initialize(const std::vector<int> &dims, int value = 1, bool random = false);
    std::ostream &operator<<(std::ostream &out, const tensor &tensor);
    tensor randn(const std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
    tensor ones(const std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
    tensor zeros(const std::vector<int> &dims, Device device = Device::cpu, bool requires_grad = false);
    tensor ones_like(const tensor &a, Device device = Device::cpu, bool requires_grad = false);
    tensor zeros_like(const tensor &a, Device device = Device::cpu, bool requires_grad = false);
}
#endif
