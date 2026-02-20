#include <stdio.h>          // Umut Tüter 1240505901
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// --- 1. Veri Tipleri ve Tensor Yapýsý ---
typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_INT8
} TensorType;

typedef struct {
    void *data;          
    uint16_t rows;       
    uint16_t cols;       
    uint32_t size;       
    TensorType dtype;    
    
    float scale;
    int32_t zero_point;
} Tensor;

// --- 2. Yardýmcý Fonksiyonlar ---
Tensor* create_tensor(uint16_t rows, uint16_t cols, TensorType type) {
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->rows = rows;
    t->cols = cols;
    t->size = rows * cols;
    t->dtype = type;
    t->scale = 1.0f;
    t->zero_point = 0;

    size_t element_size = 0;
    switch (type) {
        case DTYPE_FLOAT32: element_size = sizeof(float); break;     
        case DTYPE_FLOAT16: element_size = sizeof(uint16_t); break;  
        case DTYPE_INT8:    element_size = sizeof(int8_t); break;    
    }

    t->data = malloc(t->size * element_size);
    if (!t->data) {
        free(t);
        return NULL;
    }
    return t;
}

void free_tensor(Tensor *t) {
    if (t) {
        if (t->data) free(t->data);
        free(t);
    }
}

void set_element_f32(Tensor *t, int index, float value) {
    if (t->dtype == DTYPE_FLOAT32) {
        ((float*)t->data)[index] = value;
    } else if (t->dtype == DTYPE_INT8) {
        int32_t q_val = (int32_t)round(value / t->scale) + t->zero_point;
        if (q_val > 127) q_val = 127;
        if (q_val < -128) q_val = -128;
        ((int8_t*)t->data)[index] = (int8_t)q_val;
    }
}

float get_element_as_float(Tensor *t, int index) {
    if (index >= t->size) return 0.0f; 

    switch (t->dtype) {
        case DTYPE_FLOAT32:
            return ((float*)t->data)[index];
        case DTYPE_INT8: {
            int8_t q_val = ((int8_t*)t->data)[index];
            return (float)(q_val - t->zero_point) * t->scale;
        }
        case DTYPE_FLOAT16:
            return (float)((uint16_t*)t->data)[index]; 
        default:
            return 0.0f;
    }
}

// --- 3. Örnek Uygulama: TinyML Katmaný ---
void run_tinyml_demo() {
    int i; // Eski C standardý uyumluluðu için döngü sayacý
    
    printf("--- TinyML Gomulu Tensor Demo ---\n");
    int N = 5;

    Tensor *input = create_tensor(1, N, DTYPE_FLOAT32);
    printf("Input Tensor (F32) Bellek Boyutu: %lu bytes\n", N * sizeof(float));
    for(i=0; i<N; i++) set_element_f32(input, i, (i * 0.5f)); 

    Tensor *weights = create_tensor(1, N, DTYPE_INT8);
    weights->scale = 0.01f;     
    weights->zero_point = 0;    

    set_element_f32(weights, 0, 0.5f);  
    set_element_f32(weights, 1, -0.2f); 
    set_element_f32(weights, 2, 0.1f);
    set_element_f32(weights, 3, 0.8f);
    set_element_f32(weights, 4, -0.5f);

    float dot_product = 0.0f;
    printf("\n--- Hesaplama Basliyor ---\n");
    
    for (i = 0; i < N; i++) {
        float in_val = get_element_as_float(input, i);
        float w_val = get_element_as_float(weights, i); 
        
        printf("Index %d: Input(%.2f) * Weight(%.2f) \n", i, in_val, w_val);
        dot_product += in_val * w_val;
    }

    printf("--------------------------\n");
    printf("Sonuc (Dot Product): %.4f\n", dot_product);

    free_tensor(input);
    free_tensor(weights);
}

int main() {
    run_tinyml_demo();
    return 0;
}
