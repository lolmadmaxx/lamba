#ifndef VALUE_H
#define VALUE_H

#include <stdlib.h>
#include <stdbool.h>


#ifdef _WIN32
  #define VG_API __declspec(dllexport)
#else
  #define VG_API
#endif


// Debug and logging flags
#ifndef VG_DEBUG
#define VG_DEBUG 0
#endif

// Core Value struct and operations for automatic differentiation
typedef struct Value {
    float data;                                     // forward value
    float grad;                                     // gradient
    struct Value *left, *right;                     // children for backprop
    void (*backward)(struct Value *self);           // local backward fn
    char op;                                        // operation: '+', '*', 'r', '^', '/', '-', 'e', 'l', 't', 's', or 0
    float _exponent;                                // stored exponent for pow
    bool persistent;                                // true => do NOT free automatically (for model parameters)
} Value;

// Constructor and basic ops
VG_API Value* value_new(float x);                 
VG_API Value* value_new_persistent(float x);    
VG_API Value* value_add(Value *a, Value *b);
VG_API Value* value_mul(Value *a, Value *b);
VG_API Value* value_neg(Value *a);
VG_API Value* value_sub(Value *a, Value *b);
VG_API Value* value_pow(Value *a, float exponent);
VG_API Value* value_div(Value *a, Value *b);
VG_API Value* value_relu(Value *a);
VG_API Value* value_exp(Value *a);
VG_API Value* value_log(Value *a);
VG_API Value* value_tanh(Value *a);
VG_API Value* value_softmax(Value *a); // this is sigmoid actually

 // Memory management
VG_API void value_free(Value *v);
VG_API void value_free_graph(Value *root);
VG_API void value_backward_and_free(Value *root);  // NEW: convenient wrapper for backprop + cleanup
VG_API void value_free_graph_safe(Value *root, Value **preserve_list, int preserve_count); // Safe cleanup
VG_API void value_cleanup_constants(void);  // Clean up static constants
VG_API void value_init_constants(void);     // Initialize static constants

// Safe operations with error checking
VG_API Value* value_div_safe(Value *a, Value *b);
VG_API Value* value_log_safe(Value *a);
VG_API Value* value_pow_safe(Value *a, float exponent);

// Utility functions
VG_API float value_get_data(Value *v);
VG_API float value_get_grad(Value *v);
VG_API void value_set_data(Value *v, float data);
VG_API void value_set_grad(Value *v, float grad);
VG_API void value_zero_grad_single(Value *v);

// Backprops
VG_API void value_backward(Value *v);
VG_API void debug_print_addresses(Value *root, Value **params, int count);


// // Memory pool optimization
// typedef struct ValuePool {
//     Value *pool;
//     size_t capacity;
//     size_t next_free;
//     bool *in_use;
// } ValuePool;

// VG_API ValuePool* value_pool_new(size_t capacity);
// VG_API void value_pool_free(ValuePool *pool);
// VG_API Value* value_pool_alloc(ValuePool *pool, float data);
// VG_API void value_pool_release(ValuePool *pool, Value *v);
// VG_API void value_pool_reset(ValuePool *pool);

#endif

