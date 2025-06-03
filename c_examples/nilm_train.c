#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "../src/value.h"
#include "../src/mlp.h"

#define WINDOW 120
#define WAVE_PACKETS 2
#define FEATURE_DIM (4 * WAVE_PACKETS)

// Wave packet parameters for two inputs
static Value* A_amp[WAVE_PACKETS];
static Value* S_amp[WAVE_PACKETS];
static Value* K_amp[WAVE_PACKETS];
static Value* O_amp[WAVE_PACKETS];
static Value* X_amp[WAVE_PACKETS];

static Value* A_watt[WAVE_PACKETS];
static Value* S_watt[WAVE_PACKETS];
static Value* K_watt[WAVE_PACKETS];
static Value* O_watt[WAVE_PACKETS];
static Value* X_watt[WAVE_PACKETS];

static MLP* mlp;
static Value** mlp_params;
static int mlp_param_count;

static Value** all_params;
static int total_params;

static Value* cos_approx(Value* x) {
    Value* result = value_new_persistent(1.0f);
    Value* x2 = value_mul(x,x);
    Value* power = value_new_persistent(1.0f);
    float fact = 1.f;
    for(int n=1;n<=4;n++){
        power = value_mul(power, x2);
        fact *= (2*n-1)*(2*n);
        Value* term = value_div(power, value_new_persistent(fact));
        if(n%2==1) result = value_sub(result, term); else result = value_add(result, term);
    }
    return result;
}

static Value* sin_approx(Value* x){
    Value* result = x;
    Value* x2 = value_mul(x,x);
    Value* power = x;
    float fact = 1.f;
    for(int n=1;n<=4;n++){
        power = value_mul(power,x2);
        fact *= (2*n)*(2*n+1);
        Value* term = value_div(power, value_new_persistent(fact));
        if(n%2==1) result = value_sub(result, term); else result = value_add(result, term);
    }
    return result;
}

static void wave_forward(Value* x, Value** A, Value** S, Value** K, Value** O, Value** X, Value** out){
    for(int i=0;i<WAVE_PACKETS;i++){
        Value* sigma = value_exp(S[i]);
        Value* diff = value_sub(x, X[i]);
        Value* expn = value_div(value_mul(diff,diff), value_mul(value_mul(sigma,sigma), value_new_persistent(2.0f)));
        Value* env = value_exp(value_neg(expn));
        Value* phase = value_sub(value_mul(K[i], x), value_mul(O[i], X[i]));
        Value* cosv = cos_approx(phase);
        Value* sinv = sin_approx(phase);
        Value* base = value_mul(A[i], env);
        out[2*i] = value_mul(base, cosv);
        out[2*i+1] = value_mul(base, sinv);
    }
}

static void init_model(){
    srand(42);
    for(int i=0;i<WAVE_PACKETS;i++){
        A_amp[i]=value_new_persistent(((float)rand()/RAND_MAX)*2-1);
        S_amp[i]=value_new_persistent(((float)rand()/RAND_MAX)*2-1);
        K_amp[i]=value_new_persistent(((float)rand()/RAND_MAX)*4-2);
        O_amp[i]=value_new_persistent(((float)rand()/RAND_MAX)*4-2);
        X_amp[i]=value_new_persistent(((float)rand()/RAND_MAX)*4-2);
        A_watt[i]=value_new_persistent(((float)rand()/RAND_MAX)*2-1);
        S_watt[i]=value_new_persistent(((float)rand()/RAND_MAX)*2-1);
        K_watt[i]=value_new_persistent(((float)rand()/RAND_MAX)*4-2);
        O_watt[i]=value_new_persistent(((float)rand()/RAND_MAX)*4-2);
        X_watt[i]=value_new_persistent(((float)rand()/RAND_MAX)*4-2);
    }
    int layers[2] = {4,2};
    mlp = mlp_new(FEATURE_DIM, layers, 2);
    mlp_param_count = mlp_parameter_count(mlp);
    mlp_params = malloc(sizeof(Value*)*mlp_param_count);
    mlp_parameters(mlp, mlp_params);

    total_params = mlp_param_count + WAVE_PACKETS*5*2;
    all_params = malloc(sizeof(Value*)*total_params);
    int idx=0;
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=A_amp[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=S_amp[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=K_amp[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=O_amp[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=X_amp[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=A_watt[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=S_watt[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=K_watt[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=O_watt[i];}
    for(int i=0;i<WAVE_PACKETS;i++){all_params[idx++]=X_watt[i];}
    for(int i=0;i<mlp_param_count;i++){all_params[idx++]=mlp_params[i];}
}

static void zero_grads(){
    for(int i=0;i<total_params;i++) all_params[i]->grad=0.f;
}

static void update_params(float lr){
    for(int i=0;i<total_params;i++){
        all_params[i]->data -= lr*all_params[i]->grad;
    }
}

static void generate_window(float* amp, float* watt){
    for(int i=0;i<WINDOW;i++){
        amp[i] = (float)rand()/RAND_MAX;
        watt[i] = 0.5f*amp[i] + ((float)rand()/RAND_MAX-0.5f)*0.2f;
    }
}

int main(){
    init_model();
    const int epochs = 10;
    const float lr = 0.01f;
    float amp[WINDOW];
    float watt[WINDOW];
    for(int epoch=0; epoch<epochs; epoch++){
        generate_window(amp, watt);
        float amp_avg=0.f, watt_avg=0.f;
        for(int i=0;i<WINDOW;i++){amp_avg+=amp[i]; watt_avg+=watt[i];}
        amp_avg/=WINDOW; watt_avg/=WINDOW;

        Value* va = value_new(amp_avg);
        Value* vw = value_new(watt_avg);
        Value* feats_a[2*WAVE_PACKETS];
        Value* feats_w[2*WAVE_PACKETS];
        wave_forward(va, A_amp,S_amp,K_amp,O_amp,X_amp, feats_a);
        wave_forward(vw, A_watt,S_watt,K_watt,O_watt,X_watt, feats_w);
        Value* input[FEATURE_DIM];
        for(int i=0;i<2*WAVE_PACKETS;i++) input[i]=feats_a[i];
        for(int i=0;i<2*WAVE_PACKETS;i++) input[2*WAVE_PACKETS+i]=feats_w[i];
        Value** out = mlp_forward(mlp, input);

        Value* target1 = value_new(0.3f*amp_avg + 0.7f*watt_avg);
        Value* target2 = value_new(0.5f*amp_avg - 0.2f*watt_avg);
        Value* diff1 = value_sub(out[0], target1);
        Value* diff2 = value_sub(out[1], target2);
        Value* loss = value_add(value_mul(diff1,diff1), value_mul(diff2,diff2));
        value_backward_and_free(loss);
        update_params(lr);
        zero_grads();
        printf("Epoch %d loss %.6f\n", epoch, loss->data);
        free(out);
    }
    printf("Training complete\n");
    return 0;
}
