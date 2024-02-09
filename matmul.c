#include <stdlib.h>
#include <stdbool.h>
#include <pthread.h>
#include <unistd.h>



#define TILE 64
// A has shape (m, k)
// B has shape (k, n)
// C has shape (m, n)
 

struct gemm_args {
    size_t start, end, N, K, lda, ldb, ldc;
    float *A, *B, *C, alpha, beta;
};

void* thread_gemm_nn(void* args) {
    struct gemm_args* a = (struct gemm_args*)args;

    for (size_t ti = a->start; ti < a->end; ti += TILE) {
      for (size_t tj = 0; tj < a->N; tj += TILE) {
        for (size_t tk = 0; tk < a->K; tk += TILE) {
          size_t ti_max = ti + TILE < a->end ? ti + TILE : a->end;
          size_t tj_max = tj + TILE < a->N ? tj + TILE : a->N;
          size_t tk_max = tk + TILE < a->K ? tk + TILE : a->K;
          for (size_t i = ti; i < ti_max; i++) {
            for (size_t j = tj; j < tj_max; j++) {
              float sum = 0.0f;
              #pragma unroll
              for (size_t k = tk; k < tk_max; k++) {
                sum += a->alpha * a->A[i * a->lda + k] * a->B[k * a->ldb + j];
              }
              a->C[i * a->ldc + j] = sum + a->beta * a->C[i * a->ldc + j];
            }
          }
        }
      }
    }
    return NULL;
}

void* thread_gemm_nt(void* args){
  struct gemm_args* arg = (struct gemm_args*) args;

  for (size_t ti = arg->start; ti < arg->end; ti += TILE) {
    for (size_t tj = 0; tj < arg->N; tj += TILE) {
      for (size_t tk = 0; tk < arg->K; tk += TILE) {
        size_t ti_max = ti + TILE < arg->end ? ti + TILE : arg->end;
        size_t tj_max = tj + TILE < arg->N ? tj + TILE : arg->N;
        size_t tk_max = tk + TILE < arg->K ? tk + TILE : arg->K;
        for (size_t i = ti; i < ti_max; i++) {
          for (size_t j = tj; j < tj_max; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (size_t k = tk; k < tk_max; k++) {
              sum += arg->alpha * arg->A[i * arg->lda + k] * arg->B[j * arg->ldb + k];
            }
            arg->C[i * arg->ldc + j] = sum + arg->beta * arg->C[i * arg->ldc + j];
          }
        }
      }
    }
  } 
  return NULL;
}

void* thread_gemm_tn(void* args){
  struct gemm_args* arg = (struct gemm_args*) args;

  for(size_t ti = arg->start; ti < arg->end; ti += TILE){
    for(size_t tj = 0; tj < arg->N; tj += TILE){
      for(size_t tk = 0; tk < arg->K; tk += TILE){
        size_t ti_max = ti + TILE < arg->end ? ti + TILE : arg->end;
        size_t tj_max = tj + TILE < arg->N ? tj + TILE : arg->N;
        size_t tk_max = tk + TILE < arg->K ? tk + TILE : arg->K;
        for(size_t i = ti; i < ti_max; i++){
          for(size_t j = tj; j < tj_max; j++){
            float sum = 0.0f;
            #pragma unroll
            for(size_t k = tk; k < tk_max; k++){
              sum += arg->alpha * arg->A[k * arg->lda + i] * arg->B[k * arg->ldb + j];
            }
            arg->C[i * arg->ldc + j] = sum + arg->beta * arg->C[i * arg->ldc + j];
          }
        }
      }
    }
  }
  return NULL;
}


void* thread_gemm_tt(void* args){
  struct gemm_args* arg = (struct gemm_args*) args;

  for (size_t ti = arg->start; ti < arg->end; ti += TILE) {
    for (size_t tj = 0; tj < arg->N; tj += TILE) {
      for (size_t tk = 0; tk < arg->K; tk += TILE) {
        size_t ti_max = ti + TILE < arg->end ? ti + TILE : arg->end;
        size_t tj_max = tj + TILE < arg->N ? tj + TILE : arg->N;
        size_t tk_max = tk + TILE < arg->K ? tk + TILE : arg->K;
        for (size_t i = ti; i < ti_max; i++) {
          for (size_t j = tj; j < tj_max; j++) {
            float sum = 0.0f;
            #pragma unroll
            for (size_t k = tk; k < tk_max; k++) {
              sum += arg->alpha * arg->A[k * arg->lda + i] * arg->B[j * arg->ldb + k];
            }
            arg->C[i * arg->ldc + j] = sum + arg->beta * arg->C[i * arg->ldc + j];
          }
        }
      }
    }
  } 
  return NULL;
}

// non-transpose for both A and B
void gemm_nn(size_t M, size_t N, size_t K, size_t lda, size_t ldb, size_t ldc,
              float* A, float* B, float* C, float alpha, float beta) {
  uint16_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = M / NUM_THREADS;

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
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
              float* A, float* B, float* C, float alpha, float beta) {
  uint16_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = M / NUM_THREADS;

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
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
              float* A, float* B, float* C, float alpha, float beta) {
  uint16_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = M / NUM_THREADS;

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
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
              float* A, float* B, float* C, float alpha, float beta) {
  uint16_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t threads[NUM_THREADS];
  struct gemm_args args[NUM_THREADS];
  size_t chunk = M / NUM_THREADS;

  for (size_t i = 0; i < NUM_THREADS; i++) {
    args[i] = (struct gemm_args) {
      .start = i * chunk,
      .end = (i + 1) * chunk,
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
          float* A, float* B, float* C, float alpha, float beta) {
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
  size_t N = 2048;

  float* A = (float*)malloc(M * N * sizeof(float));
  float* B = (float*)malloc(M * N * sizeof(float));
  float* C = (float*)malloc(M * N * sizeof(float));

  for(size_t i = 0; i < M * N; i++){
    A[i] = 1.0f;
    B[i] = 1.0f;
  }
  
  gemm(false, false, M, N, N, M, N, N, A, B, C, 1.0f, 0.0f);


  return 0;
}
