// Simple inference example for ESP32
#include <math.h>
#include <stdio.h>

#define WAVE_PACKETS 3
#define FEATURE_DIM (WAVE_PACKETS*2)

// Replace values below with those generated after training
static const float A[WAVE_PACKETS] = {0};
static const float SIGMA_RAW[WAVE_PACKETS] = {0};
static const float K[WAVE_PACKETS] = {0};
static const float OMEGA[WAVE_PACKETS] = {0};
static const float X_P[WAVE_PACKETS] = {0};

// MLP parameters (49 values for architecture [6,1])
static const float MLP_PARAMS[49] = {0};

static void wave_packet_features(float x, float out[FEATURE_DIM]) {
    for(int i=0;i<WAVE_PACKETS;i++) {
        float sigma = expf(SIGMA_RAW[i]);
        float diff = x - X_P[i];
        float exponent = (diff*diff)/(2*sigma*sigma);
        float envelope = expf(-exponent);
        float phase = K[i]*x - OMEGA[i]*X_P[i];
        float cosv = cosf(phase);
        float sinv = sinf(phase);
        float base = A[i]*envelope;
        out[2*i] = base*cosv;
        out[2*i+1] = base*sinv;
    }
}

static float relu(float x){return x>0?x:0;}

static float mlp_forward(const float params[49], const float in[FEATURE_DIM]) {
    // layer1 (6 neurons)
    float h[6];
    int idx = 0;
    for(int n=0;n<6;n++) {
        float sum = params[idx + 6]; // bias
        for(int j=0;j<6;j++) sum += params[idx + j]*in[j];
        h[n] = relu(sum);
        idx += 7; // 6 weights +1 bias
    }
    // layer2 (1 neuron)
    float out = params[idx + 6];
    for(int j=0;j<6;j++) out += params[idx + j]*h[j];
    return out;
}

int main(void){
    float sample = 0.25f; // example input
    float feats[FEATURE_DIM];
    wave_packet_features(sample, feats);
    float y = mlp_forward(MLP_PARAMS, feats);
    printf("prediction: %f\n", y);
    return 0;
}
