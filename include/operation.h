#ifndef OPERATION_H
#define OPERATION_H
#include <map>
#include <vector>
#include <memory>
namespace cyg
{
    template<class T>
    class Context
    {
        std::vector<std::shared_ptr<T>> cache; // check T destructor, vector deallocates, 
    public:
        Context(){}
        std::map<std::string, int> saved_data;
        void save_for_backward(std::vector<std::shared_ptr<T>> Ts);
        std::vector<std::shared_ptr<T>> get_variables();
        Context(const Context& context):cache(context.cache), saved_data(context.saved_data){} //rule of three/five/zero
        ~Context()noexcept{
            this->saved_data.clear();
        }
    };

    template<class T>
    class Operation : public std::enable_shared_from_this<Operation<T>>
    {
        protected:
           std::unique_ptr<Context<T>> context;
        public:
            Operation(){
                this->context = std::make_unique<Context<T>>();
            };
            // @todo check guarantee that input tensors 'stay alive' for the backward pass by sharing ownership
            // may not be necessary in a single threaded env, but i suppose it will be needed to provide guarantee in a multithreaded multiprocess env
            // confirm for multithreaded multiprocess env when you start training your models/cuda integration
            // for now, use const ref to for some speed up
            virtual std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs); //for binary operator
            virtual std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs); //for unary operator
            virtual void backward(std::vector<float>* incoming_grad);
            Operation(const Operation& operation):context(operation->context){}
            ~Operation() noexcept{
                this->context.reset();
            }
    };

    template<class T>
    class Add : public Operation<T>{
        public:
            Add(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    
    template<class T>
    class Mul : public Operation<T>{
        public:
            Mul(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    template<class T>
    class Div : public Operation<T>{
        public:
            Div(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& numerator, const std::shared_ptr<T>& denominator) override;
            void backward(std::vector<float>* incoming_grad) override;
    };

    template<class T>
    class Pow : public Operation<T>{
        public:
            Pow(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& base, const std::shared_ptr<T>& exponent) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    template<class T>
    class Mean : public Operation<T>{
        public:
            Mean(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& t, const int dims, const bool keepdims=true);
            void backward(std::vector<float>* incoming_grad) override;
    };
    template<class T>
    class Exp : public Operation<T>{
        public:
            Exp(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& exponent) override;
            void backward(std::vector<float>* incoming_grad) override;
    };
    template<class T>
    class Log : public Operation<T>{
        public:
            Log(): Operation<T>(){}
            std::shared_ptr<T> forward(const std::shared_ptr<T>& v1, const char& base='e');
            void backward(std::vector<float>* incoming_grad) override;
    };
}
#endif