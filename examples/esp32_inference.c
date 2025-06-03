// Simple inference example for ESP32
#include <math.h>
#include <stdio.h>

#define WINDOW 120
#define WAVE_PACKETS 2
#define FEATURE_DIM (4 * WAVE_PACKETS)

// Replace values below with those generated after training
static const float A_AMP[WAVE_PACKETS] = {0};
static const float SIGMA_RAW_AMP[WAVE_PACKETS] = {0};
static const float K_AMP[WAVE_PACKETS] = {0};
static const float OMEGA_AMP[WAVE_PACKETS] = {0};
static const float X_P_AMP[WAVE_PACKETS] = {0};

static const float A_WATT[WAVE_PACKETS] = {0};
static const float SIGMA_RAW_WATT[WAVE_PACKETS] = {0};
static const float K_WATT[WAVE_PACKETS] = {0};
static const float OMEGA_WATT[WAVE_PACKETS] = {0};
static const float X_P_WATT[WAVE_PACKETS] = {0};

// MLP parameters for architecture [4,2]
static const float MLP_PARAMS[46] = {0};

static void wave_packet_features_single(float x,
                                        const float A[], const float SIGMA[],
                                        const float K[], const float OMEGA[],
                                        const float X_P[],
                                        float out[2 * WAVE_PACKETS]) {
    for(int i=0;i<WAVE_PACKETS;i++) {
        float sigma = expf(SIGMA[i]);
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

static void compute_features(const float amp[WINDOW], const float watt[WINDOW], float out[FEATURE_DIM]) {
    float amp_avg = 0.0f, watt_avg = 0.0f;
    for(int i=0;i<WINDOW;i++) {
        amp_avg += amp[i];
        watt_avg += watt[i];
    }
    amp_avg /= WINDOW;
    watt_avg /= WINDOW;

    wave_packet_features_single(amp_avg, A_AMP, SIGMA_RAW_AMP, K_AMP, OMEGA_AMP, X_P_AMP, out);
    wave_packet_features_single(watt_avg, A_WATT, SIGMA_RAW_WATT, K_WATT, OMEGA_WATT, X_P_WATT, &out[2*WAVE_PACKETS]);
}

static float relu(float x){return x>0?x:0;}

static void mlp_forward(const float params[46], const float in[FEATURE_DIM], float out[2]) {
    // layer1 (4 neurons)
    float h[4];
    int idx = 0;
    for(int n=0;n<4;n++) {
        float sum = params[idx + 8];
        for(int j=0;j<8;j++) sum += params[idx + j]*in[j];
        h[n] = relu(sum);
        idx += 9; // 8 weights + bias
    }
    // layer2 (2 neurons)
    for(int n=0;n<2;n++) {
        float sum = params[idx + 4];
        for(int j=0;j<4;j++) sum += params[idx + j]*h[j];
        out[n] = sum;
        idx += 5;
    }
}

int main(void){
    // Example window with dummy data
    float amp[WINDOW] = {0};
    float watt[WINDOW] = {0};
    for(int i=0;i<WINDOW;i++) {
        amp[i] = 0.2f;  // replace with real measurements
        watt[i] = 0.1f;
    }

    float feats[FEATURE_DIM];
    compute_features(amp, watt, feats);
    float out[2];
    mlp_forward(MLP_PARAMS, feats, out);
    printf("prediction: %f %f\n", out[0], out[1]);
    return 0;
}
