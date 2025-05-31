/*
karpathy's micrograd's mlp implementation in C
*/

#include "value.h"
#include "mlp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Neuron API


VG_API Neuron* neuron_new(int nin, bool nonlin) {
    Neuron *n = malloc(sizeof(Neuron));
    n->nin = nin;
    n->w = malloc(sizeof(Value*) * nin);
    for (int i = 0; i < nin; i++) {
        float r = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        n->w[i] = value_new_persistent(r);  // Use persistent for weights
    }
    n->b = value_new_persistent(0.0f);      // Use persistent for bias
    n->nonlin = nonlin;
    return n;
}

// Forward: takes array x of length nin, returns a new Value*
VG_API Value* neuron_forward(Neuron *n, Value **x) {
    Value *sum = n->b;
    for (int i = 0; i < n->nin; i++) {
        Value *prod = value_mul(n->w[i], x[i]);
        sum = value_add(sum, prod);
    }
    return n->nonlin ? value_relu(sum) : sum;
}

// Collect parameters into an external array of size nin+1
VG_API void neuron_parameters(Neuron *n, Value **out) {
    for (int i = 0; i < n->nin; i++) out[i] = n->w[i];
    out[n->nin] = n->b;
}


// Layer API

VG_API Layer* layer_new(int nin, int nout, bool nonlin) {
    Layer *L = malloc(sizeof(Layer));
    L->nout = nout;
    L->neurons = malloc(sizeof(Neuron*) * nout);
    for (int i = 0; i < nout; i++)
        L->neurons[i] = neuron_new(nin, nonlin);
    return L;
}

// Forward: returns array of Value* of length nout
VG_API Value** layer_forward(Layer *L, Value **x) {
    Value **out = malloc(sizeof(Value*) * L->nout);
    for (int i = 0; i < L->nout; i++)
        out[i] = neuron_forward(L->neurons[i], x);
    return out;
}

// Fill out_params array of size (nin+1)*nout
VG_API void layer_parameters(Layer *L, Value **out_params) {
    int idx = 0;
    for (int i = 0; i < L->nout; i++) {
        Neuron *n = L->neurons[i];
        for (int j = 0; j < n->nin; j++)
            out_params[idx++] = n->w[j];
        out_params[idx++] = n->b;
    }
}


// MLP
struct MLP {
    Layer **layers;
    int n_layers;
};

// nouts is array of length L (number of layers), each entry = number of neurons
MLP* mlp_new(int nin, int *nouts, int L) {
    MLP *M = malloc(sizeof(MLP));
    M->n_layers = L;
    M->layers = malloc(sizeof(Layer*) * L);
    int in_dim = nin;
    for (int i = 0; i < L; i++) {
        bool nonlin = (i != L - 1);
        M->layers[i] = layer_new(in_dim, nouts[i], nonlin);
        in_dim = nouts[i];
    }
    return M;
}

Value** mlp_forward(MLP *M, Value **x) {
    Value **current = x;
    Value **to_free = NULL;  // Track array to free
    
    for (int i = 0; i < M->n_layers; i++) {
        Value **next = layer_forward(M->layers[i], current);
        
        // Free previous intermediate array (not input x)
        if (to_free && to_free != x) {
            free(to_free);
        }
        
        to_free = current;  // Mark current for potential cleanup
        current = next;
    }
    
    // Don't free the final output - caller's responsibility
    return current;
}

void mlp_parameters(MLP *M, Value **out_params) {
    int idx = 0;
    for (int i = 0; i < M->n_layers; i++) {
        Layer *L = M->layers[i];
        int block = (L->neurons[0]->nin + 1) * L->nout;
        layer_parameters(L, &out_params[idx]);
        idx += block;
    }
}

// Utility: zero all gradients in a parameter array
VG_API void zero_grad(Value **params, int count) {
    for (int i = 0; i < count; i++)
        params[i]->grad = 0.0f;
}

/* —— Memory Management —— */

VG_API void neuron_free(Neuron *n) {
    if (!n) return;
    
    // Free weights array (but not individual Values - they may be shared)
    free(n->w);
    // Note: bias Value* will be cleaned up by value_free_graph if needed
    free(n);
}

VG_API void layer_free(Layer *L) {
    if (!L) return;
    
    // Free each neuron
    for (int i = 0; i < L->nout; i++) {
        neuron_free(L->neurons[i]);
    }
    free(L->neurons);
    free(L);
}

VG_API void mlp_free(MLP *M) {
    if (!M) return;
    
    // Free each layer
    for (int i = 0; i < M->n_layers; i++) {
        layer_free(M->layers[i]);
    }
    free(M->layers);
    free(M);
}

/* —— Utility Functions —— */

VG_API int mlp_parameter_count(MLP *M) {
    if (!M) return 0;
    
    int total = 0;
    for (int i = 0; i < M->n_layers; i++) {
        Layer *L = M->layers[i];
        total += layer_parameter_count(L);
    }
    return total;
}

VG_API int layer_parameter_count(Layer *L) {
    if (!L || L->nout == 0) return 0;
    
    // Each neuron has nin weights + 1 bias
    int nin = L->neurons[0]->nin;
    return (nin + 1) * L->nout;
}

VG_API void mlp_zero_grad(MLP *M) {
    if (!M) return;
    
    for (int i = 0; i < M->n_layers; i++) {
        Layer *L = M->layers[i];
        for (int j = 0; j < L->nout; j++) {
            Neuron *n = L->neurons[j];
            // Zero weight gradients
            for (int k = 0; k < n->nin; k++) {
                n->w[k]->grad = 0.0f;
            }
            // Zero bias gradient
            n->b->grad = 0.0f;
        }
    }
}

VG_API void mlp_free_forward_output(Value **output, int output_size) {
    if (output) {
        free(output);
    }
}