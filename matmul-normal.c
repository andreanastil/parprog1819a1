// code without SSE instructions

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void get_walltime(double *wct) {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double)(tp.tv_sec+tp.tv_usec/1000000.0);
}


int main() {
  // contiguous memory allocation of 2d arrays
  float* a = (float *)malloc(N*N*sizeof(float)); 
    if (a==NULL) {
      printf("Error allocating 'a'!\n");
      exit(1);
    }

  float* b = (float *)malloc(N*N*sizeof(float)); 
    if (b==NULL) {
      printf("Error allocating 'b'!\n");
      free(a);
      exit(1);
    }

  float* c = (float *)malloc(N*N*sizeof(float)); 
    if (b==NULL) {
      printf("Error allocating 'c'!\n");
      free(a);
      free(b);
      exit(1);
    }

  float sum;
  float  *pa, *pb, *pc;
  double ts, te, mflop;

  // initialization and cache warmup
  for (int i=0;i<N*N;i++) {
     a[i] = 2.0;
     b[i] = 3.0;
     c[i] = 20.0;
  } 

  //we assume that table b is already transposed 

  // get starting time
  get_walltime(&ts);
  
  pc = c;
  for (int k=0; k<N; k++){ // k stands for each row of 'a'
    pb = b; 
      for(int j=0;j<N;j++){ 
          sum = 0;
          pa = a + k * N; //get the next row of 'a' using k
          for(int i=0;i<N;i++){
              sum += (*pa) * (*pb); // matrix multiplication: A[n,j] * B[i,n]
              pa++; 
              pb++;  
          }
          *pc = sum; // storing multiplication result
          pc++; // 'c' is used consecutively 
      }
  }
  //get ending time
  get_walltime(&te);

  // checking results
  for(int i=0;i<N*N;i++){
    if(c[i] != 2.0 * 3.0 *N){
      printf("Unexpected result. c[%d] is %f instead of %f\n", i, c[i], 6.0 *N);
      break;
    }
  }

  // complexity o(n^3)
  mflop = (unsigned long)N*N*N/((te-ts)*1e6); 
  
  printf("Average array element Maccesses/sec = %f\n",mflop);

  //free arrays
  free(b);
  free(a);

  return 0;
}
