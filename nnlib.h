#ifndef NNLIB_NNLIB_H
#define NNLIB_NNLIB_H
#include <vector>
#include <random>
#include <matplot/matplot.h>
#include "third_party/cereal/archives/binary.hpp"
#include "third_party/cereal/types/vector.hpp"

inline float sgn(const float &x) {
    if (x>0){return 1.f;}
    if (std::abs(x)<=.000001){return 0.f;}
    return -1.f;
}

namespace nn {
    namespace random {
        inline std::mt19937 rng(std::random_device{}());
        inline float rnd_uniform(const float &a,const float &b) {
            std::uniform_real_distribution dist(a,b);
            return dist(rng);
        }
        inline float rnd_normal(const float &mean,const float &dev) {
            std::normal_distribution dist(mean,dev);
            return dist(rng);
        }
        inline float he_init(const float &fan_in) {
            return rnd_normal(0.f,std::sqrt(2.f/fan_in));
        }
        inline float xavier_init(const float &fan_in,const float &fan_out) {
            return rnd_normal(0.f,std::sqrt(2.f/(fan_in+fan_out)));
        }
    }
    namespace activations {
        inline float sigmoid(const float &x) {
            return 1/(1+std::exp(-x));
        }
        inline float sigmoid_deriv(const float &x) {
            return sigmoid(x)*(1.f-sigmoid(x));
        }
        inline float relu(const float &x) {
            return std::max(x,0.f);
        }
        inline float relu_deriv(const float &x) {
            return x>0.f;
        }
        inline float surrogate_relu_deriv(const float &x) {
            constexpr float a=1.67f;
            return sigmoid(a*x)+a*x*sigmoid(a*x)*(1-sigmoid(a*x));
        }
        inline std::vector<float> softmax(const std::vector<float> &x) {
        std::vector<float> exp_x;
        float exp_sum=0.f;
        for (auto &n:x) {
            exp_x.emplace_back(std::exp(n));
            exp_sum+=exp_x.back();
        }
        for (auto &n:exp_x) {
            n=n/exp_sum;
        }
        return exp_x;
    }
    }
    namespace losses {
        inline std::vector<float> mse(const std::vector<float> &pred,const std::vector<float> &real) {
            std::vector temp(pred.size(),0.f);
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp[i]+=static_cast<float>(std::pow(pred[i]-real[i],2));
            }
            return temp;
        }
        inline float mse_reduced(const std::vector<float> &pred,const std::vector<float> &real) {
            float temp=0.f;
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp+=static_cast<float>(std::pow(pred[i]-real[i],2));
            }
            return temp;
        }
        inline std::vector<float> mae(const std::vector<float> &pred,const std::vector<float> &real) {
            std::vector temp(pred.size(),0.f);
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp[i]+=std::abs(pred[i]-real[i]);
            }
            return temp;
        }
        inline float mae_reduced(const std::vector<float> &pred,const std::vector<float> &real) {
            float temp=0.f;
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp+=std::abs(pred[i]-real[i]);
            }
            return temp;
        }
        inline float categorical_crossentropy(const std::vector<float> &pred,const std::vector<float> &real) {
            float temp=0.f;
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp+=real[i]*std::log(pred[i]);
            }
            return -temp;
        }
        inline float binary_crossentropy(const std::vector<float> &pred,const std::vector<float> &real) {
            float temp=0.f;
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp+=real[i]*std::log(pred[i])+(1-real[i])*std::log(1-pred[i]);
            }
            temp/=static_cast<float>(pred.size());
            return -temp;
        }
        inline float kl_divergence(const std::vector<float> &pred,const std::vector<float> &real) {
            float temp=0.f;
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                temp+=real[i]*std::log(real[i]/pred[i]);
            }
            return temp;
        }
    }
    enum Activation_function{Linear,Relu,Sigmoid,Softmax,SurrogateRelu};
    enum Loss_function{MSE,MSE_reduced,MAE,MAE_reduced,Categorical_Crossentropy,Binary_Crossentropy,KL_Divergence};
    //the softmax dilemma:
    //  the softmax function must be applied over a layer
    //  therefore all activations must be applied at the layer level.
    struct Neuron {
        Activation_function activation{};
        float bias{};
        float bias_grad{};
        float bias_momentum{};
        float gamma{};
        float gamma_grad{};
        float gamma_momentum{};
        std::vector<float> weights{};
        std::vector<float> weight_grads{};
        std::vector<float> weight_momentum{};
        float pre_activation_value{};
        float pre_gamma_value{};
        float activation_value{};
        //activation is seperate
        void compute(const std::vector<float> &input,const bool &use_bias) {
            float temp=0.f;
            for (int i=0;i<static_cast<int>(input.size());i++) {
                temp+=input[i]*weights[i];
            }
            pre_activation_value=temp;
            if (use_bias){pre_activation_value+=bias;}
        }
        Neuron(const Activation_function &a,
            const float &b,const float &bg,const float &bm,
            const float &g,const float &gg,const float &gm,
            const std::vector<float> &w,const std::vector<float> &wg,const std::vector<float> &wm,
            const float &pav,const float &pgv,const float &av) {
            activation=a;
            bias=b;
            bias_grad=bg;
            bias_momentum=bm;
            gamma=g;
            gamma_grad=gg;
            gamma_momentum=gm;
            weights=w;
            weight_grads=wg;
            weight_momentum=wm;
            pre_activation_value=pav;
            pre_gamma_value=pgv;
            activation_value=av;
        }

        //serialization
        Neuron()=default;
        template<class Archive>
        void serialize(Archive &archive) {
            archive(activation,
                bias,bias_grad,bias_momentum,
                gamma,gamma_grad,gamma_momentum,
                weights,weight_grads,weight_momentum,
                pre_activation_value,pre_gamma_value,
                activation_value);
        }
    };
    struct Layer {
        std::vector<Neuron> neurons{};
        int neuron_count{};
        float grad_updates{};
        Activation_function activ_func{};
        bool use_bias_term{};
        Layer(const int &neurons, const Activation_function &activation_function,const bool &use_bias) {
            neuron_count=neurons;
            activ_func=activation_function;
            grad_updates=0.f;
            use_bias_term=use_bias;
        }
        //activation included.
        std::vector<float> compute(const std::vector<float> &prev_layer,const bool &normalize) {
            float norm=0.f;
            for (auto &n:neurons){
                n.compute(prev_layer,use_bias_term);
                if (normalize) {
                    norm+=static_cast<float>(std::pow(n.pre_activation_value,2));
                }
            }
            if (normalize) {
                norm/=static_cast<float>(neuron_count);
                norm+=1e-5;
                norm=std::sqrt(norm);
                for (auto &n:neurons) {
                    n.pre_gamma_value=n.pre_activation_value/norm;
                    n.pre_activation_value*=n.gamma;
                }
            }
            std::vector<float> temp;
            switch (activ_func){
                case Linear:
                    for (auto &n:neurons) {
                        n.activation_value=n.pre_activation_value;
                        temp.emplace_back(n.activation_value);
                    }
                    break;
                case Relu:
                    for (auto &n:neurons) {
                        n.activation_value=activations::relu(n.pre_activation_value);
                        temp.emplace_back(n.activation_value);
                    }
                    break;
                case Sigmoid:
                    for (auto &n:neurons) {
                        n.activation_value = activations::sigmoid(n.pre_activation_value);
                        temp.emplace_back(n.activation_value);
                    }
                    break;
                case Softmax:
                    for (auto &n:neurons) {
                        temp.emplace_back(n.pre_activation_value);
                    }
                    temp=activations::softmax(temp);
                    for (int n=0;n<neuron_count;n++) {
                        neurons[n].activation_value=temp[n];
                    }
                    break;
                case SurrogateRelu:
                    for (auto &n:neurons) {
                        n.activation_value=activations::relu(n.pre_activation_value);
                        temp.emplace_back(n.activation_value);
                    }
                    break;
                default:
                    break;
            }
            return temp;
        }
        //derivative not included.
        std::vector<float> backprop(const std::vector<float> &next_layer_error,const std::vector<float> &prev_layer_value,const bool &is_last_layer) {
            //softmax and linear activations have no deriv
            //weight gradients are grad dot x
            //bias gradients are grad dot 1 simplifying to just grad
            //input gradient (prev layer error) is grad dot w.t
            std::vector<float> grad=next_layer_error;
            grad_updates++; //number of times the layers gradients have been added to, this is for batch normalization
            switch (activ_func) {
                case (Relu):
                    for (int i=0;i<static_cast<int>(grad.size());i++) {
                        grad[i]*=activations::relu_deriv(neurons[i].pre_activation_value);
                    }
                    break;
                case (Sigmoid):
                    for (int i=0;i<static_cast<int>(grad.size());i++) {
                        grad[i]*=activations::sigmoid_deriv(neurons[i].activation_value);
                    }
                    break;
                case (Softmax):
                case (Linear):
                    break;
                case (SurrogateRelu):
                    for (int i=0;i<static_cast<int>(grad.size());i++) {
                        grad[i]*=activations::surrogate_relu_deriv(neurons[i].pre_activation_value);
                    }
                    break;
                default:
                    break;
            }
            for (int n=0;n<neuron_count;n++) {
                for (int m=0;m<static_cast<int>(prev_layer_value.size());m++) { //weight gradient
                    neurons[n].weight_grads[m]+=prev_layer_value[m]*grad[n];
                }
                if (use_bias_term) {neurons[n].bias_grad+=grad[n];} //bias gradient
                neurons[n].gamma_grad+=neurons[n].pre_gamma_value*grad[n]; //gamma gradient
            }
            if (is_last_layer){return {};}
            std::vector prev_layer_error(prev_layer_value.size(),0.f);
            for (int n=0;n<neuron_count;n++) { //previous layer error not needed in the last layer
                for (int m=0;m<static_cast<int>(prev_layer_value.size());m++) {
                    prev_layer_error[m]+=grad[n]*neurons[n].weights[m];
                }
            }
            return prev_layer_error;
        }
        Layer(const std::vector<Neuron> &n,const int &nc,const float &gu,const Activation_function &af,const bool &ubt) {
            neurons=n;
            neuron_count=nc;
            grad_updates=gu;
            activ_func=af;
            use_bias_term=ubt;
        }

        //serialization
        Layer()=default;
        template<class Archive>
        void serialize(Archive &archive) {
            archive(neurons,neuron_count,grad_updates,activ_func,use_bias_term);
        }
    };
    class Model {
        int input_layer_size{};
        std::vector<Layer> layers{};
        Loss_function model_loss{};
        bool normalize{};
    public:
        Model(const int &input_size,
            const std::vector<Layer> &layer_stack,
            const Loss_function &loss_function,
            const bool &use_RMSnorm=true
            ) {
            //defining the model
            input_layer_size=input_size;
            layers=layer_stack;
            model_loss=loss_function;
            normalize=use_RMSnorm;
            for (int i=0;i<static_cast<int>(layers.size());i++) {//for each layer
                int prev_layer_neuron_count=input_layer_size;
                if (i>0){prev_layer_neuron_count=layers[i-1].neuron_count;}
                for (int n=0;n<layers[i].neuron_count;n++) { //for each neuron
                    if (layers[i].activ_func==Relu||layers[i].activ_func==SurrogateRelu) { //he init
                        std::vector<float> weights;
                        std::vector weight_grads(prev_layer_neuron_count,0.f);
                        for (int m=0;m<prev_layer_neuron_count;m++) {
                            weights.emplace_back(random::he_init(static_cast<float>(prev_layer_neuron_count)));
                        }
                        layers[i].neurons.emplace_back(
                            layers[i].activ_func,
                            0.f,
                            0.f,
                            0.f,
                            1.f,
                            0.f,
                            0.f,
                            weights,
                            weight_grads,
                            weight_grads,
                            0.f,
                            0.f,
                            0.f
                            );
                    }
                    else {//xavier init
                        std::vector<float> weights;
                        std::vector weight_grads(prev_layer_neuron_count,0.f);
                        for (int m=0;m<prev_layer_neuron_count;m++) {
                            weights.emplace_back(random::xavier_init(static_cast<float>(prev_layer_neuron_count),static_cast<float>(layers[i].neuron_count)));
                        }
                        layers[i].neurons.emplace_back(
                            layers[i].activ_func,
                            0.f,
                            0.f,
                            0.f,
                            1.f,
                            0.f,
                            0.f,
                            weights,
                            weight_grads,
                            weight_grads,
                            0.f,
                            0.f,
                            0.f
                            );
                    }
                }
            }
        }
        std::vector<float> predict(const std::vector<float> &x) {
            std::vector<float> input=x;
            for (auto &layer:layers) {
                input=layer.compute(input,normalize);
            }
            return input;
        }
        void compute_grads(const std::vector<float> &x,const std::vector<float> &y) {
            //the grad function is going to compute the gradients for a single datapoint.
            //regularization is added here, regularization types will be a function parameter of grad_computation
            const std::vector<float> pred=predict(x);
            if (y.size()!=pred.size()){throw std::runtime_error("label and output size don't match!");}
            std::vector<float> error;
            for (int i=0;i<static_cast<int>(pred.size());i++) {
                if (model_loss==MAE||model_loss==MAE_reduced) {
                    error.emplace_back(sgn(pred[i]-y[i]));
                }
                else if (model_loss==KL_Divergence&&layers.back().activ_func==Sigmoid) {
                    error.emplace_back((pred[i]-y[i])/(pred[i]*(1-pred[i])));
                }
                else if (model_loss==MSE||model_loss==MSE_reduced) {
                    error.emplace_back(2.f*(pred[i]-y[i]));
                }
                else {
                    error.emplace_back(pred[i]-y[i]);
                }
            }
            //back
            for (int i=static_cast<int>(layers.size())-1;i>=0;i--) {
                if (i!=0) { //if its not the first layer, compute both weight and input gradients
                    std::vector prev_layer_value(layers[i-1].neuron_count,0.f);
                    for (int n=0;n<layers[i-1].neuron_count;n++) {
                        prev_layer_value[n]=layers[i-1].neurons[n].activation_value;
                    }
                    error=layers[i].backprop(error,prev_layer_value,false);
                }
                else { //otherwise compute only the weight gradients
                    error=layers[i].backprop(error,x,true);
                }
            }
        }
        void update_grads(const float &learning_rate,const float &momentum_decay=0.95f,const float clip_norm=1.0f,const float &l1_term=.0001,const float &l2_term=.001) {
            //average gradients
            for (auto &layer:layers) {
                for (auto &neuron:layer.neurons) {
                    if (layer.use_bias_term) {neuron.bias_grad/=layer.grad_updates;}
                    for (auto &weight_grad:neuron.weight_grads) {
                        weight_grad/=layer.grad_updates;
                    }
                }
            }
            //compute global gradient norm
            float global_norm=0.f;
            for (const auto &layer:layers) {
                for (const auto &neuron:layer.neurons) {
                    global_norm+=static_cast<float>(std::pow(neuron.bias_grad,2));
                    global_norm+=static_cast<float>(std::pow(neuron.gamma_grad,2));
                    for (const auto &weight_grad:neuron.weight_grads) {
                        global_norm+=static_cast<float>(std::pow(weight_grad,2));
                    }
                }
            }
            global_norm+=1e-05;
            global_norm=std::sqrt(global_norm);
            //apply global gradient norm clipping if applicable (global norm > clip norm)
            if (global_norm>clip_norm) {
                const float scaling_factor=clip_norm/global_norm;
                for (auto &layer:layers) {
                    for (auto &neuron:layer.neurons) {
                        if (layer.use_bias_term) {neuron.bias_grad*=scaling_factor;}
                        neuron.gamma_grad*=scaling_factor;
                        for (auto &weight_grad:neuron.weight_grads) {
                            weight_grad*=scaling_factor;
                        }
                    }
                }
            }
            //update parameters
            for (auto &layer:layers) {
                for (int n=0;n<layer.neuron_count;n++) {
                    for (int m=0;m<static_cast<int>(layer.neurons[n].weight_grads.size());m++) {
                        //grad compute
                        const float w_grad=layer.neurons[n].weight_grads[m]+(l1_term*sgn(layer.neurons[n].weights[m])+2*l2_term*layer.neurons[n].weights[m]);
                        //momentum compute non dragging
                        layer.neurons[n].weight_momentum[m]=(momentum_decay*layer.neurons[n].weight_momentum[m])+w_grad;
                        //nesterov momentum
                        const float nesterov_momentum=(layer.neurons[n].weight_momentum[m]*momentum_decay)+w_grad;
                        //weight compute
                        layer.neurons[n].weights[m]-=nesterov_momentum*learning_rate;
                        //zeroing
                        layer.neurons[n].weight_grads[m]=0.f; //zero grads
                    }
                    if (layer.use_bias_term) {
                        const float bias_update=layer.neurons[n].bias_grad;
                        layer.neurons[n].bias-=(bias_update+momentum_decay*layer.neurons[n].bias_momentum)*learning_rate; //normalize and apply learning rate
                        layer.neurons[n].bias_momentum=bias_update;
                        layer.neurons[n].bias_grad=0.f; //zero grads
                    }

                    const float gamma_update=layer.neurons[n].gamma_grad;
                    layer.neurons[n].gamma-=(gamma_update+momentum_decay*layer.neurons[n].gamma_momentum)*learning_rate;
                    layer.neurons[n].gamma_momentum=gamma_update;
                    layer.neurons[n].gamma_grad=0.f;

                }
                layer.grad_updates=0.f;
            }
        }
        //saving/loading
        void save(const std::string &filename) const {
            std::ofstream ofs(filename,std::ios::binary);
            cereal::BinaryOutputArchive archive(ofs);
            archive(input_layer_size,layers,model_loss,normalize);
        }
        explicit Model(const std::string &filename) { //construct a model with a saved file
            std::ifstream file(filename, std::ios::binary);
            cereal::BinaryInputArchive archive(file);
            archive(input_layer_size,layers,model_loss,normalize);
        }
        void load(const std::string &filename) { //load a model with a saved file
            std::ifstream file(filename, std::ios::binary);
            cereal::BinaryInputArchive archive(file);
            archive(input_layer_size,layers,model_loss,normalize);
        }

    };
    inline void train(Model &model, const std::string &filename,
                      std::vector<float> data_x[],std::vector<float> data_y[],
                      const int &dataset_size, const float &validation_split,
                      const int &epochs, const int &batch_size,
                      const int &plot_granularity=2, const bool &continuous_plot=true,
                      const float &learning_rate_init=.001f, const float &learning_rate_final=.0001f,
                      const float &momentum_decay_init=.95f, const float &momentum_decay_final=.8f,
                      const float &clip_norm=1.0f,const float &l1_regression_term=.0001,const float &l2_regression_term=.001
    ) {
        const int dataset_size_reg=static_cast<int>(static_cast<float>(dataset_size)*(1.f-validation_split));
        const int dataset_size_val=static_cast<int>(static_cast<float>(dataset_size)*(validation_split));
        std::vector<float> plot_x;
        std::vector<float> plot_y;
        std::vector<float> plot_y_val;
        if (continuous_plot) {
            matplot::plot(plot_x,plot_y,"-")->line_width(2.5).display_name("loss");
            matplot::hold(matplot::on);
            matplot::plot(plot_x,plot_y_val,"--")->line_width(2.5).display_name("val loss");
            matplot::hold(matplot::off);
            matplot::xlabel("Epochs");
            matplot::ylabel("Loss");
            matplot::title("Model Loss");
            matplot::legend();

        }
        for (int epoch=1;epoch<=epochs;epoch++) {
            for (int i=0;i<dataset_size_reg;i++) {
                model.compute_grads(data_x[i],data_y[i]);
                if (i%batch_size==0||i==dataset_size_reg-1) {
                    const float &momentum_decay=momentum_decay_final+(momentum_decay_init-momentum_decay_final)*(1.f-(static_cast<float>(epoch)/static_cast<float>(epochs)));
                    const float &learning_rate=learning_rate_final+(learning_rate_init-learning_rate_final)*(1.f-(static_cast<float>(epoch)/static_cast<float>(epochs)));
                    model.update_grads(learning_rate,momentum_decay,clip_norm,l1_regression_term,l2_regression_term);
                }
            }
            float loss=0.f;
            for (int i=0;i<dataset_size_reg;i++) {
                loss+=losses::mse_reduced(model.predict(data_x[i]),data_y[i]);
            }
            loss/=static_cast<float>(dataset_size_reg);
            float loss_val=0.f;
            for (int i=0;i<dataset_size_val;i++) {
                loss_val+=losses::mse_reduced(model.predict(data_x[dataset_size_reg+i]),data_y[dataset_size_reg+i]);
            }
            loss_val/=static_cast<float>(dataset_size_val);

            std::cout<<"epoch:"<<epoch<<" loss:"<<loss<<" val loss:"<<loss_val<<"\n=========================================\n";
            plot_x.emplace_back(epoch);
            plot_y.emplace_back(loss);
            plot_y_val.emplace_back(loss_val);
            if (continuous_plot&&(epoch%plot_granularity==0||epoch==1||epoch==epochs)) {
                matplot::plot(plot_x,plot_y,"-")->line_width(2.5).display_name("loss");
                matplot::hold(matplot::on);
                matplot::plot(plot_x,plot_y_val,"--")->line_width(2.5).display_name("val loss");
                matplot::hold(matplot::off);
                matplot::legend();

            }
        }
        if (!continuous_plot) {
            matplot::plot(plot_x,plot_y,"-")->line_width(2.5).display_name("loss");
            matplot::hold(matplot::on);
            matplot::plot(plot_x,plot_y_val,"--")->line_width(2.5).display_name("val loss");
            matplot::hold(matplot::off);
            matplot::xlabel("Epochs");
            matplot::ylabel("Loss");
            matplot::title("Model Loss");
            matplot::legend();
        }
        matplot::show();
        while (true) {
            std::string ans;
            std::cout<<"save model? y/n\n";
            std::cin>>ans;
            if (ans=="y"||ans=="Y") {
                model.save(filename);
                std::cout<<"model saved!\n";
                break;
            }
            if (ans=="n"||ans=="N") {
                std::cout<<"model not saved!\n";
                break;
            }
        }
    }




}


#endif //NNLIB_NNLIB_H