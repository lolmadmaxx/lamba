#ifndef MLP_H
#define MLP_H

#include "value.h"

#ifdef _WIN32
  #define VG_API __declspec(dllexport)
#else
  #define VG_API
#endif

// Struct definitions
typedef struct {
    Value **w;     // weights array
    int nin;
    Value *b;      // bias
    bool nonlin;   // some activation func
} Neuron;

typedef struct {
    Neuron **neurons;
    int nout;
} Layer;

// Forward declarations for MLP types
struct MLP;
typedef struct MLP MLP;

// MLP API
VG_API MLP* mlp_new(int nin, int *nouts, int L);
VG_API Value** mlp_forward(MLP *M, Value **x);
VG_API void mlp_parameters(MLP *M, Value **out_params);
VG_API void zero_grad(Value **params, int count);

// Memory management
VG_API void mlp_free(MLP *M);
VG_API void layer_free(Layer *L);
VG_API void neuron_free(Neuron *n);
VG_API void mlp_free_forward_output(Value **output, int output_size);

// Utility functions
VG_API int mlp_parameter_count(MLP *M);
VG_API int layer_parameter_count(Layer *L);
VG_API void mlp_zero_grad(MLP *M);

// Layer and Neuron API (exposing for more flexibility)
VG_API Neuron* neuron_new(int nin, bool nonlin);
VG_API Value* neuron_forward(Neuron *n, Value **x);
VG_API void neuron_parameters(Neuron *n, Value **out);

VG_API Layer* layer_new(int nin, int nout, bool nonlin);
VG_API Value** layer_forward(Layer *L, Value **x);
VG_API void layer_parameters(Layer *L, Value **out_params);

#endif 
