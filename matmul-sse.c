// code with SSE instructions
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <x86intrin.h>


void get_walltime(double *wct)
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  *wct = (double) (tp.tv_sec+tp.tv_usec /1000000.0);
}

int main(int argc, char** argv) {

	float *a, *b, *c;
	
	// 16 bytes alligment to perform SSE tasks
	int i = posix_memalign((void**)&a, 16, N*N*sizeof(float));
    if(i != 0) {
        exit(0);
    }
    i = posix_memalign((void**)&b, 16, N*N*sizeof(float));
    if(i != 0) {
        free(a);
        exit(0);
    }
    i = posix_memalign((void**)&c, 16, N*N*sizeof(float));
    if(i != 0) {
        free(a);
        free(b);
        exit(0);
    }

	// initialization and cache warmup
	for(int i=0; i<N*N; i++) {
    	a[i] = 2.0;
    	b[i] = 3.0;
    	c[i] = 0.0;
	}

	double ts, te, mflop;
	float *pa = a,	*pb = b, *pc = c;

	__m128 *_pa, *_pb, *_pc =(__m128 *) (pc); //special __m128 vectors 
	
	//get starting time
	get_walltime(&ts);
  
  __m128 tmp; //using a temp __m128 for holding inner loop's sum
  for(int k=0; k<N; k++) { // k stands for each row of 'a'
		_pb = (__m128*) pb;
		for(int j=0; j<N; j++) {
			_pa = (__m128 *) (pa + k * N);
			tmp = _mm_set_ps(0.0, 0.0, 0.0, 0.0);
			for(int i=0; i<N; i+=4) {
				// adding and storing per 4
				tmp= _mm_add_ps(tmp, _mm_mul_ps(*_pa, *_pb)); // tmp holds the sum of 4 multiplication results 
				_pa++; 
				_pb++;
			}
			// using SS3 _mm_hadd_ps to perform horizontal addition 	
			*_pc=_mm_hadd_ps(tmp, tmp);  
      *_pc=_mm_hadd_ps(*_pc, *_pc);	
      _mm_store_ss(pc, *_pc);	// storing the final result via _mm_store intrinsic  

			pc++; 
		}
	}
	//get ending time
	get_walltime(&te);

	// checking results
	for(int i=0; i<N*N; i++) {
		if(c [i] != 2.0*3.0*N) {
			printf("Unexpected result. c[%d] is %f instead of %f\n", i, c[i], 2.0 * 3.0 *N);
			break;
		}
	}

	// complexity o(n^3)
 	mflop = (unsigned long)N*N*N/((te-ts)*1e6);
 	printf("Average array element Maccesses/sec = %f\n",mflop);

	//free arrays
	free(a);
	free(b);
	free(c);

	return 0;
}
