#include <vector>
#include "nnlib.h"

/*
todo:
early stopping with patience
ensure nesterov momentum is applied to bias and gamma
l1 and l2 regression should be a training parameter not a model parameter
*/

int main() {
    const std::string filename="C:/Users/bryan/CLionProjects/nnlib/model.params";
    // random dataset
    constexpr int dataset_scale=5;
    std::vector<float> rnd_data_x[1024*dataset_scale]={};
    std::vector<float> rnd_data_y[1024*dataset_scale]={};
    for (int i=0;i<1024*dataset_scale;i++) {
        for (int m=0;m<32;m++) {
            rnd_data_x[i].emplace_back(nn::random::rnd_normal(0,1));
        }
        float y=7.f;
        for (int m=0;m<16;m++) {
            y+=rnd_data_x[i][m];
        }
        for (int m=16;m<32;m++) {
            y-=rnd_data_x[i][m];
        }
        int f=y;
        f%=7;
        rnd_data_y[i].emplace_back(f);
    }

    //initialize model===================
    auto model=nn::Model(
        32,
        {
            {64,nn::Relu,false},
            {64,nn::Relu,false},
            {1,nn::Linear,true}
        },
        nn::MSE,
        true
        );
    //auto model=nn::Model(filename);
    //train model========================
    nn::train(
        model,filename,rnd_data_x,rnd_data_y,1024*dataset_scale,.1f,
        250,32, 10,true,
        .01,.001,.95,.8,
        1.0,0.0001,0.001
        );


    return 0;
}