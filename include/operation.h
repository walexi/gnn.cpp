#ifndef OPERATION_H
#define OPERATION_H
#include "tensor.h"
#include <map>
#include <memory>
namespace cyg
{
    // template<typename T>
    /**
     * @todo 
     * use template specialization to reduce code here especially for math operator overloading
     */
    template <typename T>
     void op_cpu(std::vector<float>& out, const std::vector<float>& lhs, const std::vector<float>& rhs, T t);

    std::vector<float> pow(const std::vector<float>& v1, const float v2);
    std::vector<float> pow(const std::vector<float>& v1, const std::vector<float> v2);
    std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator+(const std::vector<float>& v1, const float v2);
    std::vector<float> operator-(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator-(const std::vector<float>& v1, const float v2);
    std::vector<float> operator*(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator*(const std::vector<float>& v1, const float v2);
    std::vector<float> operator/(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> operator/(const std::vector<float>& v1, const float v2);

    class Context
    {
        std::vector<std::shared_ptr<tensor>> cache; // can use unique_ptr here too,
    public:
        std::map<std::string, int> saved_data;
        void save_for_backward(std::vector<std::shared_ptr<tensor>> tensors);
        std::vector<std::shared_ptr<tensor>> get_variables();
    };

    // template<typename T>
    class Operation
    {
        protected:
           std::unique_ptr<Context> context;
        public:
            Operation(){
                this->context = std::make_unique<Context>();
            };
            virtual std::shared_ptr<tensor> forward(std::shared_ptr<tensor> lhs, std::shared_ptr<tensor> rhs);
            virtual std::shared_ptr<tensor> forward(std::shared_ptr<tensor> t);
            virtual void backward(std::vector<float>* incoming_grad);
    };

    // template<typename T>
    class Add : public Operation{
        public:
            Add(): Operation(){}
            std::shared_ptr<tensor> forward(std::shared_ptr<tensor> lhs, std::shared_ptr<tensor> rhs) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    
    // // template<typename T>
    class Mul : public Operation{
        public:
            Mul(): Operation(){}
            std::shared_ptr<tensor> forward(std::shared_ptr<tensor> lhs, std::shared_ptr<tensor> rhs) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    // // template<typename T>
    class Div : public Operation{
        public:
            Div(): Operation(){}
            std::shared_ptr<tensor> forward(std::shared_ptr<tensor> numerator, std::shared_ptr<tensor> denominator) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
}
#endif