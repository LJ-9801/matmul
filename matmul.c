#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <arm_neon.h>

#define get_ns(start, end) ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9)

#define TILE 16
// A has shape (m, k)
// B has shape (k, n)
// C has shape (m, n)
 
static inline float32_t dot(const float* a, const float* b, int n){
  float32_t sum = 0;
  int i;

  // Process elements in chunks of 4
  #pragma unroll(4)
  for(i = 0; i <= n-4; i+=4){
    float32x4_t va = vld1q_f32(a+i);
    float32x4_t vb = vld1q_f32(b+i);
    float32x4_t vc = vmulq_f32(va, vb); 
    sum += vaddvq_f32(vc);
  }

  // Process remaining elements
  float32x2_t va_low = vdup_n_f32(0);
  float32x2_t va_high = vdup_n_f32(0);
  float32x2_t vb_low = vdup_n_f32(0);
  float32x2_t vb_high = vdup_n_f32(0);

  if (n-i > 0) {
    va_low = vld1_lane_f32(a+i, va_low, 0);
    vb_low = vld1_lane_f32(b+i, vb_low, 0);
  }
  if (n-i > 1) {
    va_low = vld1_lane_f32(a+i+1, va_low, 1);
    vb_low = vld1_lane_f32(b+i+1, vb_low, 1);
  }
  if (n-i > 2) {
    va_high = vld1_lane_f32(a+i+2, va_high, 0);
    vb_high = vld1_lane_f32(b+i+2, vb_high, 0);
  }
  if (n-i > 3) {
    va_high = vld1_lane_f32(a+i+3, va_high, 1);
    vb_high = vld1_lane_f32(b+i+3, vb_high, 1);
  }

  float32x4_t va = vcombine_f32(va_low, va_high);
  float32x4_t vb = vcombine_f32(vb_low, vb_high);

  float32x4_t vc = vmulq_f32(va, vb); 
  sum += vaddvq_f32(vc);

  return sum;
}


struct gemm_args {
    size_t start, end, M, N, K, lda, ldb, ldc;
    const float *A, *B; 
    float *C, alpha, beta;
};

void* thread_gemm_nn(void* args) {
    struct gemm_args* a = (struct gemm_args*)args;

    size_t end = a->end > a->M ? a->M : a->end;

    for (size_t ti = a->start; ti < end; ti += TILE) {
      for (size_t tj = 0; tj < a->N; tj += TILE) {
        for (size_t tk = 0; tk < a->K; tk += TILE) {
          size_t ti_max = ti + TILE < a->end ? ti + TILE : end;
          size_t tj_max = tj + TILE < a->N ? tj + TILE : a->N;
          size_t tk_max = tk + TILE < a->K ? tk + TILE : a->K;
          for (size_t i = ti; i < ti_max; i++) {
            for (size_t j = tj; j < tj_max; j++) {
              float32_t sum = dot(&a->A[i * a->lda + tk], &a->B[tk * a->ldb + j], tk_max - tk); 
              a->C[i * a->ldc + j] += sum + a->beta;
            }
          }
        }
      }
    }
    return NULL;
}

void* thread_gemm_nt(void* args){
  struct gemm_args* arg = (struct gemm_args*) args;

  size_t end = arg->end > arg->M ? arg->M : arg->end;

  for (size_t ti = arg->start; ti < end; ti += TILE) {
    for (size_t tj = 0; tj < arg->N; tj += TILE) {
      for (size_t tk = 0; tk < arg->K; tk += TILE) {
        size_t ti_max = ti + TILE < arg->end ? ti + TILE : end;
        size_t tj_max = tj + TILE < arg->N ? tj + TILE : arg->N;
        size_t tk_max = tk + TILE < arg->K ? tk + TILE : arg->K;
        for (size_t i = ti; i < ti_max; i++) {
          for (size_t j = tj; j < tj_max; j++) {
            float sum = dot(&arg->A[i * arg->lda + tk], &arg->B[j * arg->lda + tk], tk_max - tk);
            arg->C[i * arg->ldc + j] += sum + arg->beta; 
          }
        }
      }
    }
  } 
  return NULL;
}

void* thread_gemm_tn(void* args){
  struct gemm_args* arg = (struct gemm_args*) args;

  size_t end = arg->end > arg->M ? arg->M : arg->end;

  for(size_t ti = arg->start; ti < end; ti += TILE){
    for(size_t tj = 0; tj < arg->N; tj += TILE){
      for(size_t tk = 0; tk < arg->K; tk += TILE){
        size_t ti_max = ti + TILE < arg->end ? ti + TILE : end;
        size_t tj_max = tj + TILE < arg->N ? tj + TILE : arg->N;
        size_t tk_max = tk + TILE < arg->K ? tk + TILE : arg->K;
        for(size_t i = ti; i < ti_max; i++){
          for(size_t j = tj; j < tj_max; j++){
            float sum = dot(&arg->A[tk * arg->ldb + i], &arg->B[tk * arg->ldb + j], tk_max - tk);
            arg->C[i * arg->ldc + j] += sum + arg->beta;
          }
        }
      }
    }
  }
  return NULL;
}


void* thread_gemm_tt(void* args){
  struct gemm_args* arg = (struct gemm_args*) args;

  size_t end = arg->end > arg->M ? arg->M : arg->end;

  for (size_t ti = arg->start; ti < end; ti += TILE) {
    for (size_t tj = 0; tj < arg->N; tj += TILE) {
      for (size_t tk = 0; tk < arg->K; tk += TILE) {
        size_t ti_max = ti + TILE < arg->end ? ti + TILE : end;
        size_t tj_max = tj + TILE < arg->N ? tj + TILE : arg->N;
        size_t tk_max = tk + TILE < arg->K ? tk + TILE : arg->K;
        for (size_t i = ti; i < ti_max; i++) {
          for (size_t j = tj; j < tj_max; j++) {
            float sum = dot(&arg->A[tk * arg->ldb + i], &arg->B[j * arg->lda + tk], tk_max - tk);
            arg->C[i * arg->ldc + j] += sum + arg->beta;
          }
        }
      }
    }
  } 
  return NULL;
}

// non-transpose for both A and B
void gemm_nn(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
              const float* A, const float* B, float* C, float alpha, float beta) {
  uint16_t MAX_NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  // thread count should be less than or equal to M
  uint16_t NUM_THREADS = M < MAX_NUM_THREADS ? M : MAX_NUM_THREADS;
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  // ceil the chunk size
  size_t chunk = (M / NUM_THREADS) + (M % NUM_THREADS != 0);

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
      .M = M,
      .N = N,
      .K = K,
      .lda = lda,
      .ldb = ldb,
      .ldc = ldc,
      .A = A,
      .B = B,
      .C = C,
      .alpha = alpha,
      .beta = beta
    };
    pthread_create(&threads[i], NULL, thread_gemm_nn, &args[i]);
  }

  for (size_t i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
}

// a non transpose, b is transpose
void gemm_nt(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
              const float* A, const float* B, float* C, float alpha, float beta) {
  uint16_t MAX_NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  // thread count should be less than or equal to M
  uint16_t NUM_THREADS = M < MAX_NUM_THREADS ? M : MAX_NUM_THREADS;
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = (M / NUM_THREADS) + (M % NUM_THREADS != 0);

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
      .M = M,
      .N = N,
      .K = K,
      .lda = lda,
      .ldb = ldb,
      .ldc = ldc,
      .A = A,
      .B = B,
      .C = C,
      .alpha = alpha,
      .beta = beta
    };
    pthread_create(&threads[i], NULL, thread_gemm_nt, &args[i]);
  }

  for (size_t i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
}

// a is transpose, b is non transpose
void gemm_tn(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
              const float* A, const float* B, float* C, float alpha, float beta) {
  uint16_t MAX_NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  // thread count should be less than or equal to M
  uint16_t NUM_THREADS = M < MAX_NUM_THREADS ? M : MAX_NUM_THREADS;
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = (M / NUM_THREADS) + (M % NUM_THREADS != 0);

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
      .M = M,
      .N = N,
      .K = K,
      .lda = lda,
      .ldb = ldb,
      .ldc = ldc,
      .A = A,
      .B = B,
      .C = C,
      .alpha = alpha,
      .beta = beta
    };
    pthread_create(&threads[i], NULL, thread_gemm_tn, &args[i]);
  }

  for (size_t i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
}

// a and b are both transpose
void gemm_tt(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
             const float* A, const float* B, float* C, float alpha, float beta) {
  uint16_t MAX_NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  // thread count should be less than or equal to M
  uint16_t NUM_THREADS = M < MAX_NUM_THREADS ? M : MAX_NUM_THREADS;
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = (M / NUM_THREADS) + (M % NUM_THREADS != 0);

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
      .M = M,
      .N = N,
      .K = K,
      .lda = lda,
      .ldb = ldb,
      .ldc = ldc,
      .A = A,
      .B = B,
      .C = C,
      .alpha = alpha,
      .beta = beta
    };
    pthread_create(&threads[i], NULL, thread_gemm_tt, &args[i]);
  }

  for (size_t i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
}

// transpose a matrix without 
// mallocing a new matrix
void transpose(size_t M, size_t N, float* A) {
  for (size_t i = 0; i < M; i++) {
    #pragma unroll
    for (size_t j = 0; j < i; j++) {
      float temp = A[i * N + j];
      A[i * N + j] = A[j * N + i];
      A[j * N + i] = temp;
    }
  }
}

void gemm(bool transA, bool transB,
          size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
          const float* A, const float* B, float* C, float alpha, float beta) {
  if (!transA && !transB) {
    gemm_nn(M, N, K, lda, ldb, ldc, A, B, C, alpha, beta);
    return;
  } else if (!transA && transB) {
    gemm_nt(M, N, K, lda, ldb, ldc, A, B, C, alpha, beta);
    return;
  } else if (transA && !transB) {
    gemm_tn(M, N, K, lda, ldb, ldc, A, B, C, alpha, beta);
    return;
  } else {
    gemm_tt(M, N, K, lda, ldb, ldc, A, B, C, alpha, beta);
    return;
  }
}

int main(){

  size_t M = 2048;
  size_t K = 2048;
  size_t N = 2048;

  float* A = (float*)malloc(M * K * sizeof(float));
  float* B = (float*)malloc(K * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  for(size_t i = 0; i < M * K; i++){
    A[i] = 1.0f;
  }

  for(size_t i = 0; i < K * N; i++){
    B[i] = 1.0f;
  }

  // time
  struct timespec start, end;

  // get time
  printf("Starting gemm_nn\n"); 
  clock_gettime(CLOCK_MONOTONIC, &start);
  gemm(false, false, M, N, K, K, N, N, A, B, C, 1.0f, 0.0f);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Finished gemm_nn with time: %lf ns\n\n", get_ns(start, end));

  printf("Starting gemm_nt\n");
  transpose(M, N, B);
  clock_gettime(CLOCK_MONOTONIC, &start);
  gemm(false, true, M, N, K, K, N, N, A, B, C, 1.0f, 0.0f); 
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Finished gemm_nt with time: %lf ns\n\n", get_ns(start, end));

  printf("Starting gemm_tn\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  gemm(true, false, M, N, K, K, N, N, A, B, C, 1.0f, 0.0f);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Finished gemm_tn with time: %lf ns\n\n", get_ns(start, end));

  printf("Starting gemm_tt\n");
  clock_gettime(CLOCK_MONOTONIC, &start);
  gemm(true, true, M, N, K, K, N, N, A, B, C, 1.0f, 0.0f);
  clock_gettime(CLOCK_MONOTONIC, &end);
  printf("Finished gemm_tt with time: %lf ns\n\n", get_ns(start, end));

  free(A);
  free(B);
  free(C); 
  
  return 0;
}
