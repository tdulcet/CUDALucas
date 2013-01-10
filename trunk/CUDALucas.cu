char program[] = "CUDALucas v2.05 Alpha";
/* CUDALucas.c
   Shoichiro Yamada Oct. 2010 

   This is an adaptation of Richard Crandall lucdwt.c, John Sweeney MacLucasUNIX.c,
   and Guillermo Ballester Valor MacLucasFFTW.c code.
   Improvement From Prime95.
   
   It also contains mfaktc code by Oliver Weihe and Eric Christenson
   adapted for CUDALucas use. Such code is under the GPL, and is noted as such.
*/


/* Include Files */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <signal.h>
#ifndef _MSC_VER
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <direct.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "cuda_safecalls.h"
#include "parse.h"

#ifdef _MSC_VER
#define strncasecmp strnicmp // _strnicmp
#endif

/* In order to have the gettimeofday() function, you need these includes on Linux:
#include <sys/time.h>
#include <unistd.h>
On Windows, you need 
#include <winsock2.h> and a definition for
int gettimeofday (struct timeval *tv, struct timezone *) {}
Both platforms are taken care of in parse.h and parse.c. */

/************************ definitions ************************************/
/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */
/* global variables needed */
double *ttmp;
double *g_ttp, *g_ttmp, *g_ttp1;
double *g_x, *g_save, *g_ct;
char *size;
int threads; 
char *g_numbits;
float *g_err;
int *g_data; 
cufftHandle plan; 

int quitting, checkpoint_iter, fftlen, s_f, t_f, r_f, d_f, k_f;
int polite, polite_f, bad_selftest=0;
int j_save;

char folder[132];
char input_filename[132], RESULTSFILE[132];
char INIFILE[132] = "CUDALucas.ini";
char AID[132]; // Assignment key
char s_residue[32];



/**************************************************************
 *
 *      FFT and other related Functions
 *
 **************************************************************/
/* rint is not ANSI compatible, so we need a definition for 
 * WIN32 and other platforms with rint.
 * Also we use that to write the trick to rint()
 */
# define RINT_x86(x) (floor(x+0.5))
# define RINT(x)  __rintd(x)
__device__ static double
__rintd (double z)
{
  double y;
asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (z));
  return (y);
}

#ifdef _MSC_VER
long long int __double2ll (double);
#endif
__device__ static long long int
__double2ll (double z)
{
  long long int y;
asm ("cvt.rni.s64.f64 %0, %1;": "=l" (y):"d" (z));
  return (y);
}

/****************************************************************************
 *           Lucas Test - specific routines                                 *
 ***************************************************************************/
__global__ void
rftfsub_kernel (int n, double *a, double *ct)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, cc, d, aj, aj1, ak, ak1;
  double new_aj, new_aj1, new_ak, new_ak1;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = threadID << 1;
  const int j2 = threadID;
  if (threadID)
    {
      int nminusj = n - j;

      wkr = 0.5 - ct[nc - j2];
      wki = ct[j2];
      aj = a[j];
      aj1 = a[1 + j];
      ak = a[nminusj];
      ak1 = a[1 + nminusj];
      xr = aj - ak;
      xi = aj1 + ak1;
      yr = wkr * xr - wki * xi;
      yi = wkr * xi + wki * xr;
      aj -= yr;
      aj1 -= yi;
      ak += yr;
      ak1 -= yi;

      new_aj1 = 2.0 * aj * aj1;
      new_aj = (aj - aj1) * (aj + aj1);

      new_ak1 = 2.0 * ak * ak1;
      new_ak = (ak - ak1) * (ak + ak1);

      xr = new_aj - new_ak;
      xi = new_aj1 + new_ak1;
      yr = wkr * xr + wki * xi;
      yi = wkr * xi - wki * xr;

      a[j] = new_aj - yr;
      a[1 + j] = yi - new_aj1;
      a[nminusj] = new_ak + yr;
      a[1 + nminusj] =  yi - new_ak1;
    }
  else
    {
      xi = a[0] - a[1];
      a[0] += a[1];
      a[1] = xi;
      a[0] *= a[0];
      a[1] *= a[1];
      a[1] = 0.5 * (a[1] - a[0]);
      a[0] += a[1];
      cc = a[0 + m];
      d = a[1 + m];
      a[1 + m] = -2.0 * cc * d;
      a[0 + m] = (cc + d) * (cc - d);
    }
}

__global__ void
normalize_kernel (double *g_in,  int *g_data, double *g_ttp, double *g_ttmp, char *g_numbits,
		  int digit, int bit, volatile float *g_err, float maxerr, int g_err_flag)
{
  long long int bigint;
  int val, numbits, mask, shifted_carry;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int index1 = blockIdx.x << 2;
  __shared__ int carry[1024 + 1];
 
  if (g_err_flag)
  {
    double tval, trint;
    float ferr;
    tval = g_in[index] * g_ttmp[index];
    trint = RINT (tval);
    ferr = tval - trint;
    ferr = fabs (ferr);
    bigint = trint;
    if (ferr > maxerr) atomicMax((int*)g_err, __float_as_int(ferr));
  }
  else bigint = __double2ll (g_in[index] * g_ttmp[index]);
  if (index == digit) bigint -= bit;

  numbits = g_numbits[index]; 
  mask = -1 << numbits;
  carry[threadIdx.x + 1] = (int) (bigint >> numbits);
  val = ((int) bigint) & ~mask;
  __syncthreads ();

  if (threadIdx.x) val += carry[threadIdx.x];
  shifted_carry = val + (1 << (numbits - 1));
  val = val - (shifted_carry & mask);
  carry[threadIdx.x] = shifted_carry >> numbits;
  if (threadIdx.x == (blockDim.x - 1))
  { if (blockIdx.x == gridDim.x - 1) g_data[2] = carry[threadIdx.x + 1] + carry[threadIdx.x];
    else   g_data[index1 + 6] =  carry[threadIdx.x + 1] + carry[threadIdx.x]; 
  }
  __syncthreads ();

  if (threadIdx.x) val += carry[threadIdx.x - 1]; 
  if (threadIdx.x > 1) g_in[index] = (double) val * g_ttp[index];
  //else g_in[index] = (double) val;
  else g_data[index1 + threadIdx.x] = val;
}

__global__ void
normalize2_kernel (double *g_x, int g_N, int threads, int *g_data, double *g_ttp1)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 2;
  const int threadID2 = threadID << 1;
  const int j = threads * threadID;
  double temp0, temp1;
  int mask, shifted_carry, numbits;

  if (j < g_N)
    {
      temp0 = g_data[threadID1] + g_data[threadID1 + 2];
      numbits = g_data[threadID1 + 3];
      mask = -1 << numbits;
      shifted_carry = temp0 + (1 << (numbits - 1)) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 = g_data[threadID1 + 1] + (shifted_carry >> numbits);
      g_x[j] = temp0 * g_ttp1[threadID2];
      g_x[j + 1] = temp1 * g_ttp1[threadID2 + 1];
    }
}

float
lucas_square (double *x, int q, int n, int iter, int last, int* offset, float* maxerr, int error_flag)
{
  int digit, bit;
  float terr = 0.0;

  *offset = (2 * *offset) % q;
  bit = (*offset + 1) % q;
  digit = floor(bit * (n / (double) q));
  bit = bit - ceil(digit * (q / (double) n));
  bit = 1 << bit;
  
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  rftfsub_kernel <<< n / 512, 128 >>> (n, g_x, g_ct);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  
  normalize_kernel <<<n / threads, threads >>> 
                    (g_x, g_data, g_ttp, g_ttmp, g_numbits, digit, bit, g_err, *maxerr, error_flag || t_f);
  normalize2_kernel <<< ((n + threads - 1) / threads + 127) / 128, 128 >>> 
                     (g_x, n, threads, g_data, g_ttp1);

  if(iter % checkpoint_iter == 0) cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
  if (error_flag)
  {
    cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
  }
  else if (polite_f && (iter % polite) == 0) cudaStreamSynchronize(0);
  return (terr);
}

/* -------- initializing routines -------- */
void
makect (int nc, double *c)
{
  int j;
  const int nch = nc >> 1;
  double delta;
  if (nc > 1)
    {
      delta = atan (1.0) / nch;
      c[0] = cos (delta * nch);
      c[nch] = 0.5 * c[0];
      for (j = 1; j < nch; j++)
	{
	  c[j] = 0.5 * cos (delta * j);
	  c[nc - j] = 0.5 * sin (delta * j);
	}
    }
}

void get_weights(int q, int n)
{
  int a, b, c, bj, j;
  double log2 = log (2.0);

  ttmp = (double *) malloc (sizeof (double) * (n));
  size = (char *) malloc (sizeof (char) * n);

  b = q % n;
  c = n - b;
  ttmp[0] = 1.0;
  bj = 0;
  for (j = 1; j < n; j++)
  {
    bj += b;
    bj %= n;
    a = bj - n;
    ttmp[j] = exp (a * log2 / n);
    size[j] = (bj >= c);
  }
  size[0] = 1;
  size[n-1] = 0;
}

void alloc_gpu_mem(int n)
{
  cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));
  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * n ));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * n / 4));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_numbits, sizeof (char) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * 2 * n / threads));
  cutilSafeCall (cudaMalloc ((void **) &g_data, sizeof (int) * 4 * n / threads));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  if (t_f) cutilSafeCall (cudaMalloc ((void **) &g_save, sizeof (double) * n));
}

void write_gpu_data(int q, int n)
{
  double *s_ttp, *s_ttmp, *s_ttp1, *s_ct;
  char *s_numbits;
  int *s_data;
  int i, j, qn = q / n;

  s_ct = (double *) malloc (sizeof (double) * (n / 4));
  s_ttp = (double *) malloc (sizeof (double) * (n));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_numbits = (char *) malloc (sizeof (char) * (n));
  s_ttp1 = (double *) malloc (sizeof (double) * 2 * (n / threads));
  s_data = (int *) malloc (sizeof (int) * (4 * n / threads));

  for (j = 0; j < n; j++)
  {
    s_ttp[j] = 1 / ttmp[j];
    s_ttmp[j] = ttmp[j] * 2.0 / n; 
    if(j % 2) s_ttmp[j] = -s_ttmp[j]; 
    s_numbits[j] = qn + size[j];
  }

  for (i = 0, j = 0; i < n; i++)
  {
    if ((i % threads) == 0)
    {
      s_ttp1[2 * j] = s_ttp[i];
      s_ttp1[2 * j + 1] = s_ttp[i + 1];
      s_data[4 * j + 3] = s_numbits[i];
      j++;
    }
  }

  makect (n / 4, s_ct);

  cudaMemcpy (g_ttmp, s_ttmp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_numbits, s_numbits, sizeof (char) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp, s_ttp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp1, s_ttp1, sizeof (double) * 2 * n / threads, cudaMemcpyHostToDevice);
  cudaMemcpy (g_data, s_data, sizeof (int) * 4 * n / threads, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ct, s_ct, sizeof (double) * (n / 4), cudaMemcpyHostToDevice);

  free ((char *) s_ct);
  free ((char *) s_ttp);
  free ((char *) s_ttmp);
  free ((char *) s_ttp1);
  free ((char *) s_data);
  free ((char *) s_numbits);
}

void init_x(double *x, int q, int n, int *offset)
{
  int j;
  int digit, bit;

  if( *offset < 0)
  {
    srand(time(0));
    *offset = rand() % q;
    bit = (*offset + 2) % q;
    digit = floor(bit * (n / (double) q));
    bit = bit - ceil(digit * (q / (double) n));
    for(j = 0; j < n; j++) x[j] = 0.0;
    x[digit] = (1 << bit) / ttmp[digit];
  }
  cudaMemcpy (g_x, x, sizeof (double) * n , cudaMemcpyHostToDevice);
}

void free_host (double *x)
{
  free ((char *) size);
  free ((char *) x);
  free ((char *) ttmp);
}

void free_gpu(void)
{
  cufftSafeCall (cufftDestroy (plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaFree ((char *) g_ttp));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_data));
  cutilSafeCall (cudaFree ((char *) g_numbits));
  if (t_f) cutilSafeCall (cudaFree ((char *) g_save));
}

void close_lucas (double *x)
{
  free_host(x);
  free_gpu();
}

void set_err_to_0(float* maxerr)
{
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  *maxerr = 0.0;
}

__global__ void
copy_kernel (double *save, double *y)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  save[threadID] = y[threadID];
}


/**************************************************************************
 *                                                                        *
 *       End LL/GPU Functions, Begin Utility/CPU Functions                *
 *                                                                        *
 **************************************************************************/

int
choose_fft_length (int q, int* index)
{  
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */
  #define COUNT 119
  int multipliers[COUNT] = {  6,     8,    12,    16,    18,    24,    32,    
                             40,    48,    64,    72,    80,    96,   120,   
                            128,   144,   160,   192,   224,   240,   256,   
                            288,   320,   336,   384,   448,   480,   512,   
                            576,   640,   672,   768,   800,   864,   896,   
                            960,  1024,  1120,  1152,  1200,  1280,  1344,
                           1440,  1536,  1600,  1680,  1728,  1792,  1920, 
                           2048,  2240,  2304,  2400,  2560,  2688,  2880,  
                           3072,  3200,  3360,  3456,  3584,  3840,  4000,  
                           4096,  4480,  4608,  4800,  5120,  5376,  5600,  
                           5760,  6144,  6400,  6720,  6912,  7168,  7680,  
                           8000,  8192,  8960,  9216,  9600, 10240, 10752, 
                          11200, 11520, 12288, 12800, 13440, 13824, 14366, 
                          15360, 16000, 16128, 16384, 17920, 18432, 19200, 
                          20480, 21504, 22400, 23040, 24576, 25600, 26880, 
                          29672, 30720, 32000, 32768, 34992, 36864, 38400,
                          40960, 46080, 49152, 51200, 55296, 61440, 65536  };
  // Largely copied from Prime95's jump tables, up to 32M
  // Support up to 64M, the maximum length with threads == 1024
  if( 0 < *index && *index < COUNT ) // override
    return 1024*multipliers[*index];  
  else if( *index >= COUNT || q == 0) 
  { /* override with manual fftlen passed as arg; set pointer to largest index <= fftlen */
    int len, i;
    for(i = COUNT - 1; i >= 0; i--)
    {
      len = 1024*multipliers[i];
      if( len <= *index )
      {
        *index = i;
        return len; /* not really necessary, but now we could decide to override ftlen with this value here */
      }
    }
  }
  else  
  { // *index < 0, not override, choose length and set pointer to proper index
    int len, i, estimate = q/20;
    for(i = 0; i < COUNT; i++)
    {
      len = 1024*multipliers[i];
      if( len >= estimate ) 
      {
        *index = i;
        return len;
      }
    }
  }
  return 0;
}

int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
  char* endptr;
  const char* ptr = str;
  int len, mult = 0;
  while( *ptr ) {
    if( *ptr == 'k' || *ptr == 'K' ) {
      mult = 1024;
      break;
    }
    if( *ptr == 'm' || *ptr == 'M' ) {
      mult = 1024*1024;
      break;
    }
    ptr++;
  }
  if( !mult ) { // No K or M, treat as before    (PS The Python else clause on loops I mention in parse.c would be useful here :) )
    mult = 1;
  }
  len = (int) strtoul(str, &endptr, 10)*mult;
  if( endptr != ptr ) { // The K or M must directly follow the num (or the num must extend to the end of the str)
    fprintf (stderr, "can't parse fft length \"%s\"\n\n", str);
    exit (2);
  }
  return len;
}

//From apsen
void
print_time_from_seconds (int sec)
{
  if (sec > 3600)
    {
      printf ("%d", sec / 3600);
      sec %= 3600;
      printf (":%02d", sec / 60);
    }
  else
    printf ("%d", sec / 60);
  sec %= 60;
  printf (":%02d", sec);
}

void
init_device (int device_number)
{
  int device_count = 0;
  cudaGetDeviceCount (&device_count);
  if (device_number >= device_count)
    {
      printf ("device_number >=  device_count ... exiting\n");
      printf ("(This is probably a driver problem)\n\n");
      exit (2);
    }
  if (d_f)
    {
      cudaDeviceProp dev;
      cudaGetDeviceProperties (&dev, device_number);
      printf ("------- DEVICE %d -------\n", device_number);
      printf ("name                %s\n", dev.name);
      printf ("totalGlobalMem      %d\n", (int) dev.totalGlobalMem);
      printf ("sharedMemPerBlock   %d\n", (int) dev.sharedMemPerBlock);
      printf ("regsPerBlock        %d\n", (int) dev.regsPerBlock);
      printf ("warpSize            %d\n", (int) dev.warpSize);
      printf ("memPitch            %d\n", (int) dev.memPitch);
      printf ("maxThreadsPerBlock  %d\n", (int) dev.maxThreadsPerBlock);
      printf ("maxThreadsDim[3]    %d,%d,%d\n",
	 dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
      printf ("maxGridSize[3]      %d,%d,%d\n", dev.maxGridSize[0],
	 dev.maxGridSize[1], dev.maxGridSize[2]);
      printf ("totalConstMem       %d\n", (int) dev.totalConstMem);
      printf ("Compatibility       %d.%d\n", dev.major, dev.minor);
      printf ("clockRate (MHz)     %d\n", dev.clockRate/1000);
      printf ("textureAlignment    %d\n", (int) dev.textureAlignment);
      printf ("deviceOverlap       %d\n", dev.deviceOverlap);
      printf ("multiProcessorCount %d\n\n", dev.multiProcessorCount);
// From Iain
    if (dev.major == 1 && dev.minor < 3)
      {
        printf
	  ("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
        exit (2);
      }
    }
  cudaSetDeviceFlags (cudaDeviceBlockingSync);
  cudaSetDevice (device_number);
}


void
rm_checkpoint (int q)
{
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c" "%d", q);
  sprintf (chkpnt_tfn, "t" "%d", q);
  (void) unlink (chkpnt_cfn);
  (void) unlink (chkpnt_tfn);
}

double *
read_checkpoint (int q, int *n, int *j, int *offset, long* total_time)
{
/* If we get file reading errors, then this doesn't try the backup t-file. Should we add that? */
  FILE *fPtr;
  int q_r, n_r, j_r, offset_r;
  long time_r;
  double *x;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
  {
      fPtr = fopen(chkpnt_tfn, "rb");
      if (!fPtr) return NULL;
  }
  // check parameters
  if (   fread(&q_r, 1, sizeof (q_r), fPtr) != sizeof (q_r) 
      || fread(&n_r, 1, sizeof (n_r), fPtr) != sizeof (n_r) 
      || fread(&j_r, 1, sizeof (j_r), fPtr) != sizeof (j_r) )
  {
      fprintf(stderr, "\nThe checkpoint appears to be corrupt. Current test will be restarted.\n");
      fclose(fPtr);
      return NULL;
  }
  if (q != q_r)
  {
      fprintf (stderr, "\nThe checkpoint doesn't match current test. Current test will be restarted.\n");
      fclose (fPtr);
      return NULL;
  }
  // check for successful read of z, delayed until here since zSize can vary
  x = (double *) malloc (sizeof (double) * n_r);
  if (fread (x, 1, sizeof (double) * (n_r), fPtr) != (sizeof (double) * (n_r)))
  {
      fprintf (stderr, "\nThe checkpoint appears to be corrupt. Current test will be restarted.\n");
      fclose (fPtr);
      free (x);
      return NULL;
  }
  /* Attempt to read total time, ignoring errors for compatibilty with old checkpoints (2.00-2.03). */
  if ( fread (&time_r, 1, sizeof(time_r), fPtr) != sizeof (time_r) ) *total_time = -1;
  else 
  {
    *total_time = time_r;
    #ifdef EBUG
    printf("Total time read from save file: ");
    print_time_from_seconds(time_r);
    printf("\n");
    #endif
  }
  if ( fread (&offset_r, 1, sizeof(offset_r), fPtr) != sizeof (offset_r) )
  {
      fprintf (stderr, "\nThe checkpoint appears to be corrupt. Current test will be restarted.\n");
      fclose (fPtr);
      return NULL;
  }
 
  // have good stuff, do checkpoint
  *n = n_r;
  *j = j_r;
  *offset = offset_r;
  fclose (fPtr);
  return x;
}

void
write_checkpoint (double *x, int q, int n, int j, int offset, long total_time)
{
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  (void) unlink (chkpnt_tfn);
  (void) rename (chkpnt_cfn, chkpnt_tfn);
  fPtr = fopen (chkpnt_cfn, "wb");
  if (!fPtr) 
  {
    fprintf(stderr, "Couldn't write checkpoint.\n");
    return;
  }
  fwrite (&q, 1, sizeof (q), fPtr);
  fwrite (&n, 1, sizeof (n), fPtr);
  fwrite (&j, 1, sizeof (j), fPtr);
  fwrite (x, 1, sizeof (double) * n, fPtr);
  fwrite (&total_time, 1, sizeof(total_time), fPtr);
  fwrite (&offset, 1, sizeof(offset), fPtr);
  #ifdef EBUG
  printf("Total time is %ld\n", total_time);
  printf("Total time again: ");
  print_time_from_seconds(total_time);
  printf("\n");
  #endif
  fclose (fPtr);
  if (s_f > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, j, s_residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, j, s_residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr) return;
      fwrite (&q, 1, sizeof (q), fPtr);
      fwrite (&n, 1, sizeof (n), fPtr);
      fwrite (&j, 1, sizeof (j), fPtr);
      fwrite (x, 1, sizeof (double) * n, fPtr);
      fwrite (&total_time, 1, sizeof(total_time), fPtr);
      fwrite (&offset, 1, sizeof(offset), fPtr);
      fclose (fPtr);
    }
}

int standardize_digits (double *x, int q, int n, int offset, int num_digits)  
{
  int j, digit, stop, qn = q / n;
  double temp, carry = 0.0;
  double lo = (double) (1 << qn);
  double hi = lo + lo;

  digit = floor(offset * (n / (double) q));
  j = (n + digit - 1) % n;
  while(RINT_x86(x[j]) == 0.0 && j != digit) j = (n + j - 1) % n;
  if(j == digit && RINT_x86(x[digit]) == 0.0) return(1);
  else if (x[j] < 0.0) carry = -1.0;
  {  
    stop = (digit + num_digits) % n;
    j = digit;
    do
    {
      x[j] = RINT_x86(x[j] * ttmp[j]) + carry;
      carry = 0.0;
      if (size[j]) temp = hi;
      else temp = lo;
      if(x[j] < -0.5)
      {
        x[j] += temp;
        carry = -1.0;
      }
      j = (j + 1) % n;
    }
    while(j != stop);
  }
  return(0);
}

void balance_digits(double* x, int q, int n)
{
  double half_low = (double) (1 << (q / n - 1));
  double low = 2.0 * half_low;
  double high = 2.0 * low;
  double upper, adj, carry = 0.0;
  int j;
 
  for(j = 0; j < n; j++)
  { 
    if(size[j])
    {
      upper = low - 0.5;
      adj = high;
    }
    else
    {
      upper = half_low - 0.5;
      adj = low;
    }
    x[j] += carry;
    carry = 0.0;
    if(x[j] > upper)
    {
      x[j] -= adj;
      carry = 1.0;
    }
    x[j] /= ttmp[j];
  }
  x[0] += carry; // Good enough for our purposes.
}

void redistribute_bits(double *old_x, char *old_size, double *new_x, int q , int old_n, int new_n)
{
  unsigned long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int mask, mask1, mask2, temp3;
  int old_qn = q / old_n;
  int new_qn = q / new_n;

  mask1 = -1 << (new_qn + 1);
  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  j = 0;
  k = 0;
  for(i = 0; i < new_n; i++)
  {
    while(k < new_qn + size[i])
    {
      temp1 = (int) old_x[j];
      temp2 += (temp1 << k);
      k += old_qn + old_size[j];
      j++;
    }
    if(size[i]) mask = mask1;
    else mask = mask2;
    temp3 = ((int) temp2) & mask;
    temp2 >>= new_qn + size[i];
    k -= new_qn + size[i];
    new_x[i] = (double) temp3;
  }
}

int
printbits (double *x, int q, int n, int offset, FILE* fp, char *expectedResidue)
{ 
  int j, k = 0;
  int digit, bit;
  unsigned long temp, residue = 0;

  j = 64 / (q / n) + 2;
  if (standardize_digits(x, q, n, offset, j))
  {
    printf ("M( %d )P, n = %dK, %s", q, n / 1024, program);
    if (fp) fprintf (fp, "M( %d )P, n = %dK, %s", q, n / 1024, program);
  }
  else
  {
    digit = floor(offset *  (n / (double) q));
    bit = offset - ceil(digit * (q / (double) n));
    j = digit;
    while(k < 64)
    {
      temp = (int) x[j];
      residue = residue + (temp << k);
      k += q / n + size[j % n];
      if(j == digit) 
      {  
         k -= bit;
         residue >>= bit;
      }
      j = (j + 1) % n;
    }
    sprintf (s_residue, "%08x%08x", (int) (residue >> 32), (int) residue);

    printf ("M( %d )C, 0x%s, n = %dK, %s", q, s_residue, n/1024, program);
    if (fp) fprintf (fp, "M( %d )C, 0x%s, n = %dK, %s", q, s_residue, n/1024, program);
    if (expectedResidue && strcmp (s_residue, expectedResidue))
    {
      bad_selftest++;
      return 1;
    }
  }
  return 0;
}

double* init_lucas(int q , int *n, int *j, int *offset, long *total_time)
{
  double *new_x, *old_x = NULL;
  int old_n, new_n, temp_n;

  *n = 0;
  *j = 1;
  *offset = -1;
  *total_time = 0;

  temp_n = fftlen; 

  old_x = read_checkpoint (q, n, j, offset, total_time);
  new_n = choose_fft_length(q, &fftlen); 

  old_n = *n;
  if(temp_n > COUNT) *n = temp_n;
  else if (!old_x || temp_n) *n = new_n;

  if(!old_x || !temp_n || ((temp_n > COUNT) && (temp_n == old_n)))
  {
    if ((*n / threads) > 65535)
  	{
  	  fprintf (stderr, "over specifications Grid = %d\n", (int) *n / threads);
  	  fprintf (stderr, "try increasing threads (%d) or decreasing FFT length (%dK)\n\n",  threads, *n / 1024);
      return NULL;
  	}
    if (q < *n)
  	{
  	  fprintf (stderr, "The prime %d is less than the fft length %d. This will cause problems.\n\n", q, *n / 1024);
      return NULL;
  	}
    if(old_x)
    { 
      new_x = old_x;
      new_n = old_n;
    }
    else new_x = (double *) malloc (sizeof (double) * new_n);
    get_weights(q, new_n);
    alloc_gpu_mem(new_n);
    write_gpu_data(q, new_n);
    init_x(new_x, q, new_n, offset);
    return new_x;
  } 
  get_weights(q, old_n);
  char *old_size = size;
  standardize_digits(old_x, q, old_n, 0, old_n);
  free ((char *) ttmp);
  get_weights(q, new_n);
  new_x = (double *) malloc (sizeof (double) * new_n );
  redistribute_bits(old_x, old_size, new_x, q, old_n, new_n);
  balance_digits(new_x, q, new_n);
  alloc_gpu_mem(new_n);
  write_gpu_data(q, new_n);
  init_x(new_x, q, new_n, offset);
  free ((char *) old_size);
  free ((char *) old_x);
  write_checkpoint (new_x, q, *n, *j - 1, *offset, *total_time);
  return new_x;
}

void
cufftbench (int cufftbench_s, int cufftbench_e, int cufftbench_d)
{
  cudaEvent_t start, stop;
  double *x;
  float outerTime;
  int i, j;
  printf ("CUFFT bench start = %d end = %d distance = %d\n", cufftbench_s,
	  cufftbench_e, cufftbench_d);

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * cufftbench_e));
  x = ((double *) malloc (sizeof (double) * cufftbench_e + 1));
  for (i = 0; i <= cufftbench_e; i++)
    x[i] = 0;
  cutilSafeCall (cudaMemcpy
		 (g_x, x, sizeof (double) * cufftbench_e,
		  cudaMemcpyHostToDevice));
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreate (&stop));
  for (j = cufftbench_s; j <= cufftbench_e; j += cufftbench_d)
    {
      cufftSafeCall (cufftPlan1d (&plan, j / 2, CUFFT_Z2Z, 1));
      cufftSafeCall (cufftExecZ2Z
		     (plan, (cufftDoubleComplex *) g_x,
		      (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 100; i++)
	cufftSafeCall (cufftExecZ2Z
		       (plan, (cufftDoubleComplex *) g_x,
			(cufftDoubleComplex *) g_x, CUFFT_INVERSE));
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      printf ("CUFFT_Z2Z size= %d time= %f msec\n", j, outerTime / 100);
      cufftSafeCall (cufftDestroy (plan));
    }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));
  free ((char *) x);
}

void
SetQuitting (int sig)
{
  quitting = 1;
 sig==SIGINT ? printf( "\tSIGINT") : (sig==SIGTERM ? printf( "\tSIGTERM") : printf( "\tUnknown signal")) ;
 printf( " caught, writing checkpoint.");
}

#ifndef _MSC_VER
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
int
_kbhit (void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr (STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr (STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl (STDIN_FILENO, F_GETFL, 0);
  fcntl (STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar ();

  tcsetattr (STDIN_FILENO, TCSANOW, &oldt);
  fcntl (STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF)
    {
      ungetc (ch, stdin);
      return 1;
    }

  return 0;
}
#else
#include <conio.h>
#endif
void interact(void); // defined below everything else
/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int
check (int q, char *expectedResidue)
{
  int n, j, last, error_flag;
  int t_ff = t_f; t_f = 1; /* Override for round off test, deactivate before actual test */
  double  *x = NULL, totalerr, avgerr;
  float maxerr, terr;
  int restarting = 0;//, continuing = 0;
  timeval time0, time1;
  long total_time = 0, start_time;
   // use start_total because every time we write a checkpoint, total_time is increased, 
   // but we don't quit everytime we write a checkpoint
  int offset;
  int new_fft = 0;

  if (!expectedResidue)
  {
    // We log to file in most cases anyway.
    signal (SIGTERM, SetQuitting);
    signal (SIGINT, SetQuitting);
  }
  do
  {				/* while (restarting) */
    totalerr = maxerr = 0.0;
    x = init_lucas (q, &n, &j, &offset, &total_time); 
    if(!x) exit (2);
    gettimeofday (&time0, NULL);
    start_time = time0.tv_sec;
    last = q - 2;		/* the last iteration done in the primary loop */
    if(j == 1)
       {
  	     printf ("Starting M%d fft length = %dK\n", q, n/1024);
  	     j_save = 0; // Only if(t_f) in old code
       }
    else if (!restarting) printf ("Continuing work from a partial result of M%d fft length = %dK iteration = %d\n", q, n/1024, j);
    restarting = 0;
    fflush (stdout);

    if( j == 1 )
    {
      printf("Running careful round off test for 1000 iterations. If average error >= 0.22, the test will restart with a larger FFT length.\n");
        /* This isn't actually any *safer* than the previous code which had error_flag = 1 if j < 1000, 
        but there would have been a *lot* of if statements to get the extra messages working, and there 
        were lots of ifs already. Now there's a lot less ifs in the main code, and now we can check the 
        average. We also reset the maxerr a lot more often (which doesn't make it *safer*, but makes the
        average more accurate). */
      double max_err = 0.0;
         
      for (; !restarting && j < 1000; j++) // Initial LL loop; no -k, no checkpoints, quitting doesn't do anything
	    {
        terr = lucas_square (x, q, n, j, last, &offset, &maxerr, 1); 
        totalerr += terr; // only works if error_flag is 1 for every iteration
	      if(terr > max_err) max_err = terr; 
	      // ^ This is necessary because we want to print the max_err every 100 iters, but we reset maxerr every 16 iters.
	      if( !(j&15) )
	      {
	        set_err_to_0(&maxerr);
	        if(terr >= 0.35)
	        {
	          /* n is not big enough; increase it and start over */
		        printf ("Iteration = %d < 1000 && err = %5.5f >= 0.35, increasing n from %dK\n", j, terr, (int) n/1024);
		        restarting = 1; 
		        fftlen++; // Whatever the user entered isn't good enough, so override
	        }
	      }
	      if( j % 100 == 0 ) 
        {
	        printf( "Iteration  %d, average error = %5.5f, max error = %5.5f\n", j, totalerr/(double)j, max_err);
	        max_err = 0.0;
	      }
	    } // End special/initial LL for loop; now we determine if the error is acceptable
      if( !restarting ) 
      {
        avgerr = totalerr/(j-1); // j should be 1000, but we only did 999 iters
        if( avgerr >= 0.22 ) 
        {
          printf("Iteration 1000, average error = %5.5f >= 0.25 (max error = %5.5f), increasing FFT length and restarting\n", avgerr, max_err);
          fftlen++; // Whatever the user entered isn't good enough, so override
          restarting = 1;
        } 
        else if( avgerr < 0 ) 
        {
          fprintf(stderr, "Something's gone terribly wrong! Avgerr = %5.5f < 0 !\n", avgerr);
        } 
        else 
        {
          printf("Iteration 1000, average error = %5.5f < 0.25 (max error = %5.5f), continuing test.\n", avgerr, max_err);
          set_err_to_0(&maxerr);
        }
      }
    } // End special/initial testing      
      
    if( !t_ff ) t_f = 0; // Undo t_f override from beginning of check()

    for (; !restarting && j <= last; j++) // Main LL loop, j should be >= 1000 unless user killed a test with j<1000
    {
	    if ((j % 100) == 50) error_flag = 1;
	    else error_flag = 0;
      terr = lucas_square (x, q, n, j, last, &offset, &maxerr, error_flag);
      if (error_flag)
	    { 
	      if (terr >= 0.35)
		    {
		      if (t_f)
	        {
		        gettimeofday(&time1, NULL);
		        printf ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, writing checkpoint file (because -t is enabled) and exiting.\n\n", j, (double) terr, (int) n/1024);
		        cutilSafeCall (cudaMemcpy (x, g_save, sizeof (double) * n, cudaMemcpyDeviceToHost));
		        if( total_time >= 0) {total_time += (time1.tv_sec - start_time);}
		        write_checkpoint (x, q, n, j_save + 1, offset, total_time); 
		      }
		      printf ("Iteration = %d >= 1000 && err = %5.5g >= 0.35, fft length = %dK, restarting from last checkpoint with increased fft length.\n\n", j, (double) terr, (int) n/1024);
          fftlen++;
          restarting = 1;
          new_fft = 1;
	      }
	      else		// error_flag && terr < 0.35
	      {
		      if (t_f)
		      {
		        copy_kernel <<< n / 128, 128 >>> (g_save, g_x);
		        j_save = j;
		      }
		    }
	    }  
	    if ((j % checkpoint_iter) == 0 || quitting)
      {
	      if(quitting) cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
	      gettimeofday (&time1, NULL);
	      if(!expectedResidue)
        {
          if( total_time >= 0 ) 
          {
	          total_time += (time1.tv_sec - start_time);
	          start_time = time1.tv_sec;
	        }
	        write_checkpoint (x, q, n, j + 1, offset, total_time); 
        }
	      if(!quitting)
        {
          printf ("Iteration %d ", j);
          int ret = printbits (x, q, n, offset, 0, expectedResidue);
	        long diff = time1.tv_sec - time0.tv_sec;
	        long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	        long diff2 = (long) ((last - j) / checkpoint_iter * (diff1 / 1e6));
	        gettimeofday (&time0, NULL);
	        printf (" err = %5.5f (", maxerr);
	        print_time_from_seconds (diff);
	        printf (" real, %4.4f ms/iter, ETA ", diff1 / 1000.0 / checkpoint_iter);
	        print_time_from_seconds (diff2);
	        printf (")\n");
	        fflush (stdout);
	        set_err_to_0(&maxerr); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
          if (expectedResidue) 
	        {
            j = last + 1;
		        fftlen = 0;
            if(ret) printf("Expected residue [%s] does not match actual residue [%s]\n\n", expectedResidue, s_residue);
	          else printf("This residue is correct.\n\n");
	        }
          if(new_fft)
          {  
              printf("Sticking point passed. Swithcing back to old fft length.\n");
              fftlen--;
              restarting = 1;
              new_fft = 0;
          }
	      }
	      else
        {
	        if(total_time >= 0) 
          {
	          printf(" Estimated time spent so far: ");
	          print_time_from_seconds(total_time);
	        }
	        printf("\n\n");
		      j = last + 1;
	      }
	    }
      if ( k_f && !quitting && !expectedResidue && (!(j & 15)) && _kbhit() ) interact(); // abstracted to clean up check()
	    fflush (stdout);	    
	  } /* end main LL for-loop */
	
    if (!restarting && !expectedResidue && !quitting)
	  { // done with test
	    gettimeofday (&time1, NULL);
	    FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	    if(!fp) 
	    {
	      fprintf (stderr, "Cannot write results to %s\n\n", RESULTSFILE);
	      exit (1);
	    }
	    cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
	    printbits (x, q, n, offset, fp, 0); 
	    if( total_time >= 0 ) 
            { /* Only print time if we don't have an old checkpoint file */
	      total_time += (time1.tv_sec - start_time);
	      printf (", estimated total time = ");
	      print_time_from_seconds(total_time);
	    }
	  
	    if( AID[0] && strncasecmp(AID, "N/A", 3) )  
            { // If (AID is not empty), AND (AID is NOT "N/A") (case insensitive)
              fprintf(fp, ", AID: %s\n", AID);
	    } 
            else
              fprintf(fp, "\n");
	    unlock_and_fclose(fp);
	    fflush (stdout);
	    rm_checkpoint (q);
	    printf("\n\n");
	  }
    close_lucas (x);
  }
  while (restarting);
  return (0);
}

void parse_args(int argc, char *argv[], int* q, int* device_numer, 
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d);
		/* The rest of the opts are global */
int main (int argc, char *argv[])
{ 
  printf("\n");
  quitting = 0;
#define THREADS_DFLT 256
#define CHECKPOINT_ITER_DFLT 10000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define T_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define POLITE_DFLT 1
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"
  
  /* "Production" opts to be read in from command line or ini file */
  int q = -1;
  int device_number = -1, f_f = 0;
  checkpoint_iter = -1;
  threads = -1;
  fftlen = -1;
  s_f = t_f = d_f = k_f = -1;
  polite_f = polite = -1;
  AID[0] = input_filename[0] = RESULTSFILE[0] = 0; /* First character is null terminator */
  char fft_str[132] = "\0";
  
  /* Non-"production" opts */
  r_f = 0;
  int cufftbench_s, cufftbench_e, cufftbench_d;  
  cufftbench_s = cufftbench_e = cufftbench_d = 0;

  parse_args(argc, argv, &q, &device_number, &cufftbench_s, &cufftbench_e, &cufftbench_d);
  /* The rest of the args are globals */
  
  if (file_exists(INIFILE))
  {  
   if( checkpoint_iter < 1 && 		!IniGetInt(INIFILE, "CheckpointIterations", &checkpoint_iter, CHECKPOINT_ITER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
   if( threads < 1 && 			!IniGetInt(INIFILE, "Threads", &threads, THREADS_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: %d\n", THREADS_DFLT);
   if( s_f < 0 && 			!IniGetInt(INIFILE, "SaveAllCheckpoints", &s_f, S_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
   if( 		     	     s_f > 0 && !IniGetStr(INIFILE, "SaveFolder", folder, SAVE_FOLDER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
   if( t_f < 0 && 			!IniGetInt(INIFILE, "CheckRoundoffAllIterations", &t_f, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option CheckRoundoffAllIterations; using default: off\n");
   if( polite < 0 && 			!IniGetInt(INIFILE, "Polite", &polite, POLITE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT);
   if( k_f < 0 && 			!IniGetInt(INIFILE, "Interactive", &k_f, 0) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
   if( device_number < 0 &&		!IniGetInt(INIFILE, "DeviceNumber", &device_number, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n");
   if( d_f < 0 &&			!IniGetInt(INIFILE, "PrintDeviceInfo", &d_f, D_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
   if( !input_filename[0] &&		!IniGetStr(INIFILE, "WorkFile", input_filename, WORKFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
    /* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
   if( !RESULTSFILE[0] && 		!IniGetStr(INIFILE, "ResultsFile", RESULTSFILE, RESULTSFILE_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
   if( fftlen < 0 && 			!IniGetStr(INIFILE, "FFTLength", fft_str, "\0") )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
  }
  else // no ini file
    {
      fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");
      if( checkpoint_iter < 1 ) checkpoint_iter = CHECKPOINT_ITER_DFLT;
      if( threads < 1 ) threads = THREADS_DFLT;
      if( fftlen < 0 ) fftlen = 0;
      if( s_f < 0 ) s_f = S_F_DFLT;
      if( t_f < 0 ) t_f = T_F_DFLT;
      if( k_f < 0 ) k_f = K_F_DFLT;
      if( device_number < 0 ) device_number = 0;
      if( d_f < 0 ) d_f = D_F_DFLT;
      if( polite < 0 ) polite = POLITE_DFLT;
      if( !input_filename[0] ) sprintf(input_filename, WORKFILE_DFLT);
      if( !RESULTSFILE[0] ) sprintf(RESULTSFILE, RESULTSFILE_DFLT);
  }
  
  if( fftlen < 0 ) { // possible if -f not on command line
      fftlen = fft_from_str(fft_str);
  }
  if (polite == 0) {
    polite_f = 0;
    polite = 1;
  } else {
    polite_f = 1;
  }
  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
  {
    fprintf(stderr, "Error: thread count is invalid.\n");
    fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
    exit(2);
  }
  f_f = fftlen; // if the user has given an override... then note this length must be kept between tests
    
  
  init_device (device_number);

  if (r_f)
    {
      fftlen = 0;
      checkpoint_iter = 10000;
      t_f = 1;
      check (86243, "23992ccd735a03d9");
      check (132049, "4c52a92b54635f9e");
      check (216091, "30247786758b8792");
      check (756839, "5d2cbe7cb24a109a");
      check (859433, "3c4ad525c2d0aed0");
      check (1257787, "3f45bf9bea7213ea");
      check (1398269, "a4a6d2f0e34629db");
      check (2976221, "2a7111b7f70fea2f");
      check (3021377, "6387a70a85d46baf");
      check (6972593, "88f1d2640adb89e1");
      check (13466917, "9fdc1f4092b15d69");
      check (20996011, "5fc58920a821da11");
      check (24036583, "cbdef38a0bdc4f00");
      check (25964951, "62eb3ff0a5f6237c");
      check (30402457, "0b8600ef47e69d27");
      check (32582657, "02751b7fcec76bb1");
      check (37156667, "67ad7646a1fad514");
      check (42643801, "8f90d78d5007bba7");
      check (43112609, "e86891ebf6cd70c4");
      if (bad_selftest)
      {
        fprintf(stderr, "Error: There ");
        bad_selftest > 1 ? fprintf(stderr, "were %d bad selftests!\n",bad_selftest) 
        		 : fprintf(stderr, "was a bad selftest!\n");
      }
    }
  else if (cufftbench_d)
    cufftbench (cufftbench_s, cufftbench_e, cufftbench_d);
  else
    {
      if (s_f)
	{
#ifndef _MSC_VER
	  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
	  if (mkdir (folder, mode) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#else
	  if (_mkdir (folder) != 0)
	    fprintf (stderr,
		     "mkdir: cannot create directory `%s': File exists\n",
		     folder);
#endif
	}
      if (q <= 0)
        {
          int error;
	
	  #ifdef EBUG
	  printf("Processed INI file and console arguments correctly; about to call get_next_assignment().\n");
	  #endif
	  do 
            { // while(!quitting)
	
	                 
	      fftlen = f_f; // fftlen and AID change between tests, so be sure to reset them
	      AID[0] = 0;
	                   
  	      error = get_next_assignment(input_filename, &q, &fftlen, &AID);
               /* Guaranteed to write to fftlen ONLY if specified on workfile line, so that if unspecified, the pre-set default is kept. */
	      if( error > 0) exit (2); // get_next_assignment prints warning message	  
	      #ifdef EBUG
	      printf("Gotten assignment, about to call check().\n");
	      #endif
              check (q, 0);
	  
	      if(!quitting) // Only clear assignment if not killed by user, i.e. test finished 
	        {
	          error = clear_assignment(input_filename, q);
	          if(error) exit (2); // prints its own warnings
	        }
	    
	    } 
          while(!quitting);  
      }
    else // Exponent passed in as argument
      {
	if (!valid_assignment(q, fftlen)) {printf("\n");} //! v_a prints warning
	else check (q, 0);
      }
  } // end if(-r) else if(-cufft) else(workfile)
} // end main()

void parse_args(int argc, char *argv[], int* q, int* device_number, 
		int* cufftbench_s, int* cufftbench_e, int* cufftbench_d)
{
while (argc > 1)
    {
      if (strcmp (argv[1], "-t") == 0)
	{
	  t_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-h") == 0)
        {
      	  fprintf (stderr,
	       "$ CUDALucas -h|-v\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-t] [-polite iteration] [-k] exponent|input_filename\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] -cufftbench start end distance\n\n");
	  fprintf (stderr,
	       "                       -h          print this help message\n");
	  fprintf (stderr,
	       "                       -v          print version number\n");
	  fprintf (stderr,
	       "                       -info       print device information\n");
	  fprintf (stderr,
	       "                       -i          set .ini file name (default = \"CUDALucas.ini\")\n");
      	  fprintf (stderr,
	       "                       -threads    set threads number (default = 256)\n");
      	  fprintf (stderr,
	       "                       -f          set fft length (if round off error then exit)\n");
      	  fprintf (stderr,
	       "                       -s          save all checkpoint files\n");
      	  fprintf (stderr,
	       "                       -t          check round off error all iterations\n");
      	  fprintf (stderr,
	       "                       -polite     GPU is polite every n iterations (default -polite 1) (-polite 0 = GPU aggressive)\n");
      	  fprintf (stderr,
	       "                       -cufftbench exec CUFFT benchmark (Ex. $ ./CUDALucas -d 1 -cufftbench 1179648 6291456 32768 )\n");
      	  fprintf (stderr, 
      	       "                       -r          exec residue test.\n");
      	  fprintf (stderr,
	       "                       -k          enable keys (p change -polite, t disable -t, s change -s)\n\n");
      	  exit (2);          
      	}
      else if (strcmp (argv[1], "-v") == 0)
        {  
          printf("%s\n\n", program);
          exit (2);
        }
      else if (strcmp (argv[1], "-polite") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -polite option\n\n");
	      exit (2);
	    }
	  polite = atoi (argv[2]);
	  if (polite == 0)
	    {
	      polite_f = 0;
	      polite = 1;
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-r") == 0)
	{
	  r_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-k") == 0)
	{
	  k_f = 1;
	  argv++;
	  argc--;
	}
      else if (strcmp (argv[1], "-d") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d option\n\n");
	      exit (2);
	    }
	  *device_number = atoi (argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-i") == 0)
	{
	  if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -i option\n\n");
	      exit (2);
	    }
	  sprintf (INIFILE, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-info") == 0)
        {
          d_f = 1;
          argv++;
          argc--;
        }
      else if (strcmp (argv[1], "-cufftbench") == 0)
	{
	  if (argc < 5 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-')
	    {
	      fprintf (stderr, "can't parse -cufftbench option\n\n");
	      exit (2);
	    }
	  *cufftbench_s = atoi (argv[2]);
	  *cufftbench_e = atoi (argv[3]);
	  *cufftbench_d = atoi (argv[4]);
	  argv += 4;
	  argc -= 4;
	}
      else if (strcmp (argv[1], "-threads") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threads option\n\n");
	      exit (2);
	    }
	  threads = atoi (argv[2]);
	  if (threads != 32 && threads != 64 && threads != 128
	      && threads != 256 && threads != 512 && threads != 1024)
	    {
	      fprintf(stderr, "Error: thread count is invalid.\n");
	      fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-c") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  checkpoint_iter = atoi (argv[2]);
	  if (checkpoint_iter == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-f") == 0)
	{
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -f option\n\n");
	      exit (2);
	    }
	  fftlen = fft_from_str(argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else if (strcmp (argv[1], "-s") == 0)
	{
	  s_f = 1;
	  if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -s option\n\n");
	      exit (2);
	    }
	  sprintf (folder, "%s", argv[2]);
	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  if (*q != -1 || strcmp (input_filename, "") != 0 )
	    {
	      fprintf (stderr, "can't parse options\n\n");
	      exit (2);
	    }
	  int derp = atoi (argv[1]);
	  if (derp == 0) {
	    sprintf (input_filename, "%s", argv[1]);
	  } else { *q = derp; }
	  argv++;
	  argc--;
	}
    }
}

void interact(void)
{
  int c = getchar ();
  if (c == 'p')
    if (polite_f)
	  {
	    polite_f = 0;
	    printf ("   -polite 0\n");
	  }
    else
     {
      polite_f = 1;
      printf ("   -polite %d\n", polite);
     }
  else if (c == 't')
    {
      t_f = 0;
      printf ("   disabling -t\n");
    }
  else if (c == 's')
    if (s_f == 1)
      {
        s_f = 2;
        printf ("   disabling -s\n");
      }
    else if (s_f == 2)
      {
        s_f = 1;
        printf ("   enabling -s\n");
      }
  fflush (stdin);
}
