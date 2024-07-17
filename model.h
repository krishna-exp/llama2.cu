
#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#define BATCH_SIZE 38
#define SCRATCH_BUFFER_SIZE 32

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wqT; // (layer, dim, n_heads * head_size)
    float* wkT; // (layer, dim, n_kv_heads * head_size)
    float* wvT; // (layer, dim, n_kv_heads * head_size)
    float* woT; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1T; // (layer, hidden_dim, dim)
    float* w2T; // (layer, dim, hidden_dim)
    float* w3T; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    float *d_raw;
    float *d_rffw;
    float *d_wqT;
    float *d_wkT;
    float *d_wvT;
    float *d_woT;
    float *d_w1T;
    float *d_w2T;
    float *d_w3T;
} DeviceBuffers;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (BATCH_SIZE, dim,)
    float *xb; // same, but inside a residual branch (BATCH_SIZE, dim,)
    float *xb2; // an additional buffer just for convenience (BATCH_SIZE, dim,)
    float *hb; // buffer for hidden dimension in the ffn (BATCH_SIZE, hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (BATCH_SIZE, hidden_dim,)
    float *q; // query (BATCH_SIZE, dim,)
    float *k; // key (BATCH_SIZE, dim,)
    float *v; // value (BATCH_SIZE, dim,)
    float *att; // buffer for scores/attention values (BATCH_SIZE, n_heads, seq_len)
    float *logits; // output logits (BATCH_SIZE, vocab_size)
    // kv cache
    float* key_cache;   // (BATCH_SIZE, layer, seq_len, dim)
    float* value_cache; // (BATCH_SIZE, layer, seq_len, dim)
    float *d_sb; // ((BATCH_SIZE, n_heads, SCRATCH_BUFFER_SIZE) + BATCH_SIZE)

    // common buffers
    float *d_rfw;
    float *d_wclsT;

    // compute buffer
    DeviceBuffers cb;

    // memory buffer
    DeviceBuffers mb;
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    float* data; // memory mapped data pointer
} Transformer;


#endif // MODEL_H_INCLUDED
