#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <inttypes.h>

/* op implementations and their backward helpers  */

static void backward_add(Value *self) {
    self->left->grad  += 1.0f * self->grad;
    self->right->grad += 1.0f * self->grad;
}

static void backward_mul(Value *self) {
    self->left->grad  += self->right->data * self->grad;
    self->right->grad += self->left->data  * self->grad;
}

static void backward_pow(Value *self) {
    float e = self->_exponent;
    self->left->grad += e * powf(self->left->data, e - 1.0f) * self->grad;
}

static void backward_relu(Value *self) {
    float grad_through = (self->left->data > 0.0f ? 1.0f : 0.0f);
    // printf("backward_relu: self=%p left=%p self->grad=%f grad_through=%f\n", (void*)self, (void*)self->left, self->grad, grad_through);
    self->left->grad += grad_through * self->grad;
}

static void backward_exp(Value *self) {
    // d/dx exp(x) = exp(x), and exp(x) is stored in self->data
    self->left->grad += self->data * self->grad;
}

static void backward_log(Value *self) {
    // d/dx log(x) = 1/x
    self->left->grad += (1.0f / self->left->data) * self->grad;
}

static void backward_tanh(Value *self) {
    // d/dx tanh(x) = 1 - tanh²(x), and tanh(x) is stored in self->data
    float tanh_val = self->data;
    self->left->grad += (1.0f - tanh_val * tanh_val) * self->grad;
}

static void backward_softmax(Value *self) {
    // For the current implementation which is actually sigmoid: exp(x) / (1 + exp(x))
    // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    // sigmoid(x) is stored in self->data
    float sigmoid_val = self->data;
    self->left->grad += sigmoid_val * (1.0f - sigmoid_val) * self->grad;
}



/*  constructor  */

VG_API Value* value_new(float x) {
    Value *v = malloc(sizeof(Value));
    if (!v) { perror("malloc"); exit(1); }
    v->data      = x;
    v->grad      = 0.0f;
    v->left      = v->right = NULL;
    v->backward  = NULL;
    v->op         = 0;
    v->_exponent  = 0.0f;
    v->persistent = false;  // transient by default
    return v;
}

VG_API Value* value_new_persistent(float x) {
    Value *v = malloc(sizeof(Value));
    if (!v) { perror("malloc"); exit(1); }
    v->data      = x;
    v->grad      = 0.0f;
    v->left      = v->right = NULL;
    v->backward  = NULL;
    v->op         = 0;
    v->_exponent  = 0.0f;
    v->persistent = true;   // persistent - won't be auto-freed
    return v;
}

/* forward ops  */

VG_API Value* value_add(Value *a, Value *b) {
    Value *z = value_new(a->data + b->data);
    z->left     = a;  z->right = b;
    z->op        = '+';
    z->backward  = backward_add;
    return z;
}

VG_API Value* value_mul(Value *a, Value *b) {
    Value *z = value_new(a->data * b->data);
    z->left     = a;  z->right = b;
    z->op        = '*';
    z->backward  = backward_mul;
    return z;
}

VG_API Value* value_pow(Value *a, float exponent) {
    Value *z = value_new(powf(a->data, exponent));
    z->left      = a;
    z->op         = '^';
    z->backward   = backward_pow;
    z->_exponent  = exponent;
    return z;
}

/* Static constants to avoid memory leaks */
static Value* neg_one = NULL;

/* Initialize constants once */
static void init_constants() {
    if (!neg_one) {
        neg_one = value_new_persistent(-1.0f);  // Make constants persistent
    }
}

VG_API Value* value_neg(Value *a) {
    init_constants();
    return value_mul(a, neg_one);
}

VG_API Value* value_sub(Value *a, Value *b) {
    init_constants();
    return value_add(a, value_mul(b, neg_one));
}

VG_API Value* value_div(Value *a, Value *b) {
    Value *inv_b = value_pow(b, -1.0f);
    return value_mul(a, inv_b);
}

VG_API Value* value_div_safe(Value *a, Value *b) {
    if (fabsf(b->data) < 1e-7f) {
        if (VG_DEBUG) {
            fprintf(stderr, "Warning: Division by zero or near-zero value (%.2e) detected!\n", b->data);
        }
        // Return a very large value instead of crashing
        return value_new(a->data > 0 ? 1e6f : -1e6f);
    }
    return value_mul(a, value_pow(b, -1.0f));
}

VG_API Value* value_relu(Value *a) {
    Value *z = value_new(a->data < 0.0f ? 0.0f : a->data);
    z->left     = a;
    z->op        = 'r';
    z->backward  = backward_relu;
    return z;
}

VG_API Value* value_exp(Value *a) {
    Value *z = value_new(expf(a->data));
    z->left     = a;
    z->op        = 'e';
    z->backward  = backward_exp;
    return z;
}

VG_API Value* value_log(Value *a) {
    Value *z = value_new(logf(a->data));
    z->left     = a;
    z->op        = 'l';
    z->backward  = backward_log;
    return z;
}

VG_API Value* value_log_safe(Value *a) {
    if (a->data <= 0.0f) {
        if (VG_DEBUG) {
            fprintf(stderr, "Warning: Logarithm of non-positive value (%.2e) detected!\n", a->data);
        }
        return value_new(logf(1e-7f));
    }
    return value_log(a);
}

VG_API Value* value_pow_safe(Value *a, float exponent) {
    // Check for potential overflow or underflow
    if (fabsf(a->data) > 1e3f && exponent > 10.0f) {
        if (VG_DEBUG) {
            fprintf(stderr, "Warning: Potential overflow in pow(%.2e, %.2e)\n", a->data, exponent);
        }
        return value_new(a->data > 0 ? 1e10f : -1e10f);
    }
    if (a->data == 0.0f && exponent < 0.0f) {
        if (VG_DEBUG) {
            fprintf(stderr, "Warning: Division by zero in pow(0, %.2e)\n", exponent);
        }
        return value_new(1e10f);
    }
    return value_pow(a, exponent);
}

VG_API Value* value_tanh(Value *a) {
    Value *z = value_new(tanhf(a->data));
    z->left     = a;
    z->op        = 't';
    z->backward  = backward_tanh;
    return z;
}

VG_API Value* value_softmax(Value *a) {
    // This is actually a sigmoid implementation: exp(x) / (1 + exp(x))
    // True softmax would require a vector of values
    float exp_a = expf(a->data);
    Value *z = value_new(exp_a / (1.0f + exp_a));
    z->left     = a;
    z->op        = 's';  // 's' for softmax (actually sigmoid)
    z->backward  = backward_softmax;
    return z;
}

/* topological sort backprop */

typedef struct Node {
    Value *v;
    struct Node *next;
} Node;

/* Helper function for recursive memory cleanup */
static void _value_free_recursive(Value *v, int *seen, size_t mask);

/* DFS building post-order list */
static void build_topo(Value *v, Node **list, int *seen, size_t mask) {
    size_t idx = ((size_t)v) & mask;
    if (seen[idx]) return;
    seen[idx] = 1;
    if (v->left)  build_topo(v->left,  list, seen, mask);
    if (v->right) build_topo(v->right, list, seen, mask);
    Node *n = malloc(sizeof(Node));
    n->v    = v;
    n->next = *list;
    *list   = n;
}

static void zero_grad_topo(Value *v, int *seen, size_t mask) {
    size_t idx = ((size_t)v) & mask;
    if (seen[idx]) return;
    seen[idx] = 1;
    v->grad = 0.0f;
    if (v->left)  zero_grad_topo(v->left,  seen, mask);
    if (v->right) zero_grad_topo(v->right, seen, mask);
}

VG_API void value_backward(Value *root) {
    /* zero all grads in the graph */
    size_t buckets = 1<<16;
    int *seen = calloc(buckets, sizeof(int));
    zero_grad_topo(root, seen, buckets - 1);
    free(seen);

    root->grad = 1.0f;  // seed

    /* build topo list */
    Node *topo = NULL;
    buckets = 1<<16;
    seen = calloc(buckets, sizeof(int));
    build_topo(root, &topo, seen, buckets - 1);
    free(seen);

    /* apply backward in post-order */
    for (Node *n = topo; n; n = n->next) {
        if (n->v->backward) n->v->backward(n->v);
    }

    /* cleanup */
    while (topo) {
        Node *tmp = topo;
        topo = topo->next;
        free(tmp);
    }
}

static void print_param_addresses(Value **params, int count) {
    // printf("Parameter addresses:\n");
    for (int i = 0; i < count; i++) {
        // printf("  param[%d]: %p\n", i, (void*)params[i]);
    }
}

static void print_graph_addresses(Value *v, int *seen, size_t mask) {
    size_t idx = ((size_t)v) & mask;
    if (seen[idx]) return;
    seen[idx] = 1;
    // printf("Graph node: %p\n", (void*)v);
    if (v->left)  print_graph_addresses(v->left,  seen, mask);
    if (v->right) print_graph_addresses(v->right, seen, mask);
}

// Add a debug function to call from Python
VG_API void debug_print_addresses(Value *root, Value **params, int count) {
    size_t buckets = 1<<16;
    int *seen = calloc(buckets, sizeof(int));
    print_param_addresses(params, count);
    print_graph_addresses(root, seen, buckets - 1);
    free(seen);
}

/* Memory Management  */

VG_API void value_free(Value *v) {
    if (!v) return;
    free(v);
}

VG_API void value_free_graph(Value *root) {
    if (!root) return;
    
    size_t buckets = 1<<16;
    int *seen = calloc(buckets, sizeof(int));
    _value_free_recursive(root, seen, buckets - 1);
    free(seen);
}

VG_API void value_backward_and_free(Value *root) {
    if (!root) return;
    
    // First, run backpropagation
    value_backward(root);
    
    // Then, free non-persistent nodes in the computation graph
    value_free_graph(root);
}

static void _value_free_recursive(Value *v, int *seen, size_t mask) {
    if (!v) return;
    size_t idx = ((size_t)v) & mask;
    if (seen[idx]) return;
    seen[idx] = 1;
    
    // Don't free persistent nodes (model parameters)
    if (v->persistent) {
        return;
    }
    
    if (v->left) _value_free_recursive(v->left, seen, mask);
    if (v->right) _value_free_recursive(v->right, seen, mask);
    free(v);
}

/* Safe recursive cleanup that preserves specific nodes */
static bool is_in_preserve_list(Value *v, Value **preserve_list, int preserve_count) {
    for (int i = 0; i < preserve_count; i++) {
        if (preserve_list[i] == v) return true;
    }
    return false;
}

static void _value_free_safe_recursive(Value *v, int *seen, size_t mask, Value **preserve_list, int preserve_count) {
    if (!v) return;
    size_t idx = ((size_t)v) & mask;
    if (seen[idx]) return;
    seen[idx] = 1;
    
    // Don't free if this node is in the preserve list
    if (is_in_preserve_list(v, preserve_list, preserve_count)) {
        return;
    }
    
    if (v->left) _value_free_safe_recursive(v->left, seen, mask, preserve_list, preserve_count);
    if (v->right) _value_free_safe_recursive(v->right, seen, mask, preserve_list, preserve_count);
    free(v);
}

VG_API void value_free_graph_safe(Value *root, Value **preserve_list, int preserve_count) {
    if (!root) return;
    
    size_t buckets = 1<<16;
    int *seen = calloc(buckets, sizeof(int));
    _value_free_safe_recursive(root, seen, buckets - 1, preserve_list, preserve_count);
    free(seen);
}

VG_API void value_cleanup_constants(void) {
    if (neg_one) {
        free(neg_one);
        neg_one = NULL;
    }
}

VG_API void value_init_constants(void) {
    init_constants();
}

/* Utility Functions  */

VG_API float value_get_data(Value *v) {
    return v ? v->data : 0.0f;
}

VG_API float value_get_grad(Value *v) {
    return v ? v->grad : 0.0f;
}

VG_API void value_set_data(Value *v, float data) {
    if (v) v->data = data;
}

VG_API void value_set_grad(Value *v, float grad) {
    if (v) v->grad = grad;
}

VG_API void value_zero_grad_single(Value *v) {
    if (v) v->grad = 0.0f;
}

// /* Memory Pool Implementation */

// VG_API ValuePool* value_pool_new(size_t capacity) {
//     ValuePool *pool = malloc(sizeof(ValuePool));
//     if (!pool) return NULL;
    
//     pool->pool = malloc(sizeof(Value) * capacity);
//     pool->in_use = calloc(capacity, sizeof(bool));
//     pool->capacity = capacity;
//     pool->next_free = 0;
    
//     if (!pool->pool || !pool->in_use) {
//         free(pool->pool);
//         free(pool->in_use);
//         free(pool);
//         return NULL;
//     }
    
//     return pool;
// }

// VG_API void value_pool_free(ValuePool *pool) {
//     if (!pool) return;
//     free(pool->pool);
//     free(pool->in_use);
//     free(pool);
// }

// VG_API Value* value_pool_alloc(ValuePool *pool, float data) {
//     if (!pool) return NULL;
    
//     // Find next free slot
//     for (size_t i = 0; i < pool->capacity; i++) {
//         size_t idx = (pool->next_free + i) % pool->capacity;
//         if (!pool->in_use[idx]) {
//             pool->in_use[idx] = true;
//             pool->next_free = (idx + 1) % pool->capacity;
            
//             Value *v = &pool->pool[idx];
//             v->data = data;
//             v->grad = 0.0f;
//             v->left = v->right = NULL;
//             v->backward = NULL;
//             v->op = 0;
//             v->_exponent = 0.0f;
//             v->persistent = false;  // Pool values are transient by default
            
//             return v;
//         }
//     }
    
//     // Pool is full, fall back to regular allocation
//     if (VG_DEBUG) {
//         fprintf(stderr, "Warning: Value pool exhausted, falling back to malloc\n");
//     }
//     return value_new(data);
// }

// VG_API void value_pool_release(ValuePool *pool, Value *v) {
//     if (!pool || !v) return;
    
//     // Check if value is from this pool
//     if (v >= pool->pool && v < pool->pool + pool->capacity) {
//         size_t idx = v - pool->pool;
//         pool->in_use[idx] = false;
//     }
//     // If not from pool, it was allocated with malloc, so we should free it
//     else {
//         free(v);
//     }
// }

// VG_API void value_pool_reset(ValuePool *pool) {
//     if (!pool) return;
    
//     for (size_t i = 0; i < pool->capacity; i++) {
//         pool->in_use[i] = false;
//     }
//     pool->next_free = 0;
// }
