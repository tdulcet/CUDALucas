char program[] = "CUDALucas v2.05 Beta";
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
double *ttmp, *ttp;
double *g_ttmp, *g_ttp1, *g_ttp2;
double *g_x, *g_save, *g_ct;
char *size;
float *g_err;
int *g_data, *g_carry;
cufftHandle plan;
cudaDeviceProp dev;

int multipliers[250];
int fft_count;
//int threads, threads1 = 256, threads2 = 128;
int threads[3] = {256,128,128};
int max_threads;
int error_reset = 85;
int quitting, checkpoint_iter, fftlen, s_f, t_f, r_f, d_f, k_f;
int polite, polite_f;//, bad_selftest=0;

char folder[132];
char input_filename[132], RESULTSFILE[132];
char INIFILE[132] = "CUDALucas.ini";
char AID[132]; // Assignment key
char s_residue[32];

__constant__ double g_ttpinc[2];

__constant__ int g_qn[1];


void set_ttpinc(double *hblah){
    cudaMemcpyToSymbol(g_ttpinc, hblah, 2 * sizeof(double));
}

void set_qn(int *h_qn){
    cudaMemcpyToSymbol(g_qn, h_qn, 1 * sizeof(int));
}


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

/****************************************************************************
 *           Lucas Test - specific routines                                 *
 ***************************************************************************/
__global__ void
rftfsub_kernel (int n, double *a, double *ct)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  double wkr, wki, xr, xi, yr, yi, aj, aj1, ak, ak1;
  const int m = n >> 1;
  const int nc = n >> 2;
  const int j = threadID << 1;
  const int nminusj = n - j;

  if (threadID)
    {
      wkr = 0.5 - ct[nc - threadID];
      wki = ct[threadID];
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
      xi = 2.0 * aj * aj1;
      xr = (aj - aj1) * (aj + aj1);
      yi = 2.0 * ak * ak1;
      yr = (ak - ak1) * (ak + ak1);
      aj = xr - yr;
      aj1 = xi + yi;
      ak = wkr * aj + wki * aj1;
      ak1 = wkr * aj1 - wki * aj;
      a[j] = xr - ak;
      a[1 + j] = ak1 - xi;
      a[nminusj] = yr + ak;
      a[1 + nminusj] =  ak1 - yi;
    }
  else
    {
      aj = a[0];
      aj1 = a[1];
      xi = aj - aj1;
      aj += aj1;
      xi *= xi;
      aj *= aj;
      aj1 = 0.5 * (xi - aj);
      a[0] = aj + aj1;
      a[1] = aj1;
      xr = a[0 + m];
      xi = a[1 + m];
      a[1 + m] = -2.0 * xr * xi;
      a[0 + m] = (xr + xi) * (xr - xi);
    }
}

__global__ void norm1 (double *g_in,
                         int *g_data,
                         double *g_ttmp,
                         int *g_carry,
		                     volatile float *g_err,
		                     float maxerr,
		                     int digit,
		                     int bit)
{
  long long int bigint[2];
  int val[2], numbits[2] = {g_qn[0],g_qn[0]}, mask[2], shifted_carry;
  double ttp_temp;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  const int index1 = blockIdx.x << 1;
  __shared__ int carry[1024 + 1];

  {
    double tval[2], trint[2];
    float ferr[2];

    tval[0] = g_ttmp[index];
    ttp_temp = g_ttmp[index + 1];
    trint[0] = g_in[index];
    trint[1] = g_in[index + 1];
    if(tval[0] < 0.0)
    {
      numbits[0]++;
      tval[0] = -tval[0];
    }
    if(ttp_temp < 0.0)
    {
      numbits[1]++;
      ttp_temp = -ttp_temp;
    }
    tval[1] = tval[0] * g_ttpinc[numbits[0] == g_qn[0]];
    tval[0] = trint[0] * tval[0];
    tval[1] = trint[1] * tval[1];
    trint[0] = RINT (tval[0]);
    ferr[0] = tval[0] - trint[0];
    ferr[0] = fabs (ferr[0]);
    bigint[0] = (long long int) trint[0];
    trint[1] = RINT (tval[1]);
    ferr[1] = tval[1] - trint[1];
    ferr[1] = fabs (ferr[1]);
    bigint[1] = (long long int) trint[1];
    mask[0] = -1 << numbits[0];
    mask[1] = -1 << numbits[1];
    if(ferr[0] < ferr[1]) ferr[0] = ferr[1];
    if(ferr[0] > maxerr) atomicMax((int*) g_err, __float_as_int(ferr[0]));
  }
  if(index == digit) bigint[0] -= bit;
  if((index + 1) == digit) bigint[1] -= bit;
  val[1] = ((int) bigint[1]) & ~mask[1];
  carry[threadIdx.x + 1] = (int) (bigint[1] >> numbits[1]);
  val[0] = ((int) bigint[0]) & ~mask[0];
  val[1] += (int) (bigint[0] >> numbits[0]);
  __syncthreads ();

  if (threadIdx.x) val[0] += carry[threadIdx.x];
  shifted_carry = val[1] - (mask[1] >> 1);
  val[1] = val[1] - (shifted_carry & mask[1]);
  carry[threadIdx.x] = shifted_carry >> numbits[1];
  shifted_carry = val[0] - (mask[0] >> 1);
  val[0] = val[0] - (shifted_carry & mask[0]);
  val[1] += shifted_carry >> numbits[0];
  __syncthreads ();

  if (threadIdx.x == (blockDim.x - 1))
  {
    if (blockIdx.x == gridDim.x - 1) g_carry[0] = carry[threadIdx.x + 1] + carry[threadIdx.x];
    else   g_carry[blockIdx.x + 1] =  carry[threadIdx.x + 1] + carry[threadIdx.x];
  }

  if (threadIdx.x)
  {
    val[0] += carry[threadIdx.x - 1];
    {
        g_in[index + 1] = (double) val[1] * ttp_temp;
        ttp_temp *= -g_ttpinc[numbits[0] == g_qn[0]];
        g_in[index] = (double) val[0] * ttp_temp;
    }
  }
  else
  {
    g_data[index1] = val[0];
    g_data[index1 + 1] = val[1];
  }
}

__global__ void norm2 (double *g_x,
                         int g_N,
                         int threads,
                         int *g_data,
                         int *g_carry,
                         double *g_ttp1)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads * threadID) << 1;
  int temp0, temp1;
  int mask, shifted_carry, numbits= g_qn[0];
  double temp;

  if (j < g_N)
  {
    temp0 = g_data[threadID1] + g_carry[threadID];
    temp1 = g_data[threadID1 + 1];
    temp = g_ttp1[threadID];
    if(temp < 0.0)
    {
      numbits++;
      temp = -temp;
    }
    mask = -1 << numbits;
    shifted_carry = temp0 - (mask >> 1) ;
    temp0 = temp0 - (shifted_carry & mask);
    temp1 += (shifted_carry >> numbits);
    g_x[j + 1] = temp1 * temp;
    temp *= -g_ttpinc[numbits == g_qn[0]];
    g_x[j] = temp0 * temp;
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
  rftfsub_kernel <<< n / (4 * threads[1]), threads[1] >>> (n, g_x, g_ct);
  cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  norm1 <<<n / (2 * threads[0]), threads[0] >>> (g_x, g_data, g_ttmp, g_carry, g_err, *maxerr, digit, bit);
  norm2 <<< (n / (2 * threads[0]) + threads[2] - 1) / threads[2], threads[2] >>> (g_x, n, threads[0], g_data, g_carry, g_ttp1);

  if (error_flag)
  {
    cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
  }
  else if (polite_f && (iter % polite) == 0) cutilSafeThreadSync();
  return (terr);
}

/* -------- initializing routines -------- */
void
makect (int nc, double *c)
{
  int j;
  double d = (double) (nc << 1);
  for (j = 1; j < nc; j++) c[j] = 0.5 * cospi (j / d);
}

void get_weights(int q, int n)
{
  int a, b, c, bj, j;

  ttmp = (double *) malloc (sizeof (double) * (n));
  ttp = (double *) malloc (sizeof (double) * (n));
  size = (char *) malloc (sizeof (char) * n);

  b = q % n;
  c = n - b;
  ttmp[0] = 1.0;
  ttp[0] = 1.0;
  bj = 0;
  for (j = 1; j < n; j++)
  {
    bj += b;
    bj %= n;
    a = bj - n;
    ttmp[j] = exp2 (a / (double) n);
    ttp[j] = exp2 (-a / (double) n);
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
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp2, sizeof (double) * 63 * n / 2048));
  cutilSafeCall (cudaMalloc ((void **) &g_data, sizeof (int) * n / 32));//threads));
  cutilSafeCall (cudaMalloc ((void **) &g_carry, sizeof (int) * n / 64));//2 / threads));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  if (t_f) cutilSafeCall (cudaMalloc ((void **) &g_save, sizeof (double) * n));
}

void write_gpu_data(int q, int n)
{
  double *s_ttp, *s_ttmp, *s_ttp1, *s_ct;
  int i, j, k, l = 0;
  int qn = q / n, b = q % n;;
  double *h_ttpinc;
  int *h_qn;

  s_ct = (double *) malloc (sizeof (double) * (n / 4));
  s_ttp = (double *) malloc (sizeof (double) * (n));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_ttp1 = (double *) malloc (sizeof (double) * 63 * n / 2048);
	h_ttpinc = (double*) malloc(2 * sizeof(double));
	h_qn = (int*) malloc(1 * sizeof(int));

  for (j = 0; j < n; j++)
  {
    s_ttp[j] = ttp[j];
    if(j % 2 == 0) s_ttmp[j] = ttmp[j] * 2.0 / n;
    else s_ttmp[j] = ttp[j];
    if(size[j]) s_ttmp[j] = -s_ttmp[j];
  }

  h_ttpinc[0] = -exp2((b - n) / (double) n);
  h_ttpinc[1] = -exp2(b / (double) n);
  set_ttpinc(h_ttpinc);
  h_qn[0] = qn;
  set_qn(h_qn);

  for(k = 32; k <= 1024; k *= 2)
  {
    for (i = 0, j = 0; i < n ; i += 2 * k)
    {
      s_ttp1[l + j] = s_ttp[i + 1];
      if(size[i]) s_ttp1[l + j] = -s_ttp1[l + j];
      j++;
    }
    l += j;
  }

  makect (n / 4, s_ct);

  cudaMemcpy (g_ttmp, s_ttmp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp2, s_ttp1, sizeof (double) * 63 * n / 2048, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ct, s_ct, sizeof (double) * (n / 4), cudaMemcpyHostToDevice);

  l = 0;
  k = 32;
  while(k < threads[0])
  {
    l += n / (2 * k);
    k *= 2;
  }
  g_ttp1 = &g_ttp2[l];

  free ((char *) s_ct);
  free ((char *) s_ttp);
  free ((char *) s_ttmp);
  free ((char *) s_ttp1);
  free ((char *) h_ttpinc);
  free ((char *) h_qn);
  free ((char *) ttp);
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
  cutilSafeCall (cudaFree ((char *) g_ttp2));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_data));
  cutilSafeCall (cudaFree ((char *) g_carry));
  if (t_f) cutilSafeCall (cudaFree ((char *) g_save));
}

void close_lucas (double *x)
{
  free_host(x);
  free_gpu();
}

void reset_err(float* maxerr, float value)
{
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  *maxerr *= value;
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
void init_threads(int n)
{
  FILE *threadf;
  char buf[132];
  char threadfile[32];
  //int no_file = 0, no_entry = 1;
  int th1 = 0, th2 = 0, th3 = 0;
  int temp_n;

  sprintf (threadfile, "%s threads.txt", dev.name);
  threadf = fopen(threadfile, "r");
  if(threadf)
  {
    while(fgets(buf, 132, threadf) != NULL)
    {
      sscanf(buf, "%d %d %d %d", &temp_n, &th1, &th2, &th3);
      if(n == temp_n * 1024)
      {
        threads[0] = th1;
        threads[1] = th2;
        threads[2] = th3;
        //no_entry = 0;
      }
    }
    fclose(threadf);
  }
  //else no_file = 1;
  //if(no_file || no_entry)
  //{
  //  if(no_file) printf("No %s file found. Using default thread sizes.\n", threadfile);
  //  else if(no_entry) printf("No entry for fft = %dk found. Using default thread sizes.\n", n / 1024);
  //  printf("For optimal thread selection, please run\n");
  //  printf("./CUDAPm1 -cufftbench %d %d r\n", n / 1024, n / 1024);
  //  printf("for some small r, 0 < r < 6 e.g.\n");
  //  fflush(NULL);
  //}
  return;
}

int init_ffts()
{
  //#define COUNT 139
  FILE *fft;
  char buf[132];
  int next_fft, j = 0, i = 0;
  int first_found = 0;
  #define COUNT 160
  int default_mult[COUNT] = {  //this batch from GTX570 timings
                                2,     8,    10,    14,    16,    18,    20,    32,    36,    42,
                               48,    50,    56,    60,    64,    70,    80,    84,    96,   112,
                              120,   126,   128,   144,   160,   162,   168,   180,   192,   224,
                              256,   288,   320,   324,   336,   360,   384,   392,   400,   448,
                              512,   576,   640,   648,   672,   720,   768,   784,   800,   864,
                              896,   900,  1024,  1152,  1176,  1280,  1296,  1344,  1440,  1568,
                             1600,  1728,  1792,  2048,  2160,  2304,  2352,  2592,  2688,  2880,
                             3024,  3136,  3200,  3584,  3600,  4096,  4320,  4608,  4704,  5120,
                             5184,  5600,  5760,  6048,  6144,  6272,  6400,  6480,  7168,  7200,
                             7776,  8064,  8192,  8640,  9216,  9408, 10240, 10368, 10584, 10800,
                            11200, 11520, 12096, 12288, 12544, 12960, 13824, 14336, 14400, 16384,
                            17496, 18144, 19208, 19600, 20000, 20250, 21952, 23328, 23814, 24300,
                            24500, 25088, 25600, 26244, 27000, 27216, 28000, 28672, 31104, 31250,
                            32000, 32400, 32768, 33614, 34992, 36000, 36288, 38416, 39200, 39366,
                            40500, 41472, 42336, 43200, 43904, 47628, 49000, 50000, 50176, 51200,
                            52488, 54432, 55296, 56000, 57344, 60750, 62500, 64000, 64800, 65536 };

  char fftfile[32];

  sprintf (fftfile, "%s fft.txt", dev.name);
  fft = fopen(fftfile, "r");
  if(!fft)
  {
    //printf("No %s file found. Using default fft lengths.\n", fftfile);
    //printf("For optimal fft selection, please run\n");
    //printf("./CUDAPm1 -cufftbench 1 8192 r\n");
    //printf("for some small r, 0 < r < 6 e.g.\n");
    //fflush(NULL);
    for(j = 0; j < COUNT; j++) multipliers[j] = default_mult[j];
  }
  else
  {
    while(fgets(buf, 132, fft) != NULL)
    {
      int le = 0;

      sscanf(buf, "%d", &le);
      if(next_fft = atoi(buf))
      {
        if(!first_found)
        {
          while(i < COUNT && default_mult[i] < next_fft)
          {
            multipliers[j] = default_mult[i];
            i++;
            j++;
          }
          multipliers[j] = next_fft;
          j++;
          first_found = 1;
        }
        else
        {
          multipliers[j] = next_fft;
          j++;
        }
      }
    }
    while(default_mult[i] < multipliers[j - 1] && i < COUNT) i++;
    while(i < COUNT)
    {
      multipliers[j] = default_mult[i];
      j++;
      i++;
    }
    fclose(fft);
  }
  return j;
}

int
choose_fft_length (int q, int *index)
{
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */

  if( 0 < *index && *index < fft_count ) return 1024*multipliers[*index];
  else if( *index >= fft_count || q == 0)
  { /* override with manual fftlen passed as arg; set pointer to largest index <= fftlen */
    int len, i;
    for(i = fft_count - 1; i >= 0; i--)
    {
      len = 1024*multipliers[i];
      if( len <= *index )
      {
        *index = i;
        return len; /* not really necessary, but now we could decide to override fftlen with this value */
      }
    }
  }
  else
  { // *index < 0, not override, choose length and set pointer to proper index
    int i;
    int estimate = ceil(0.0000358738168878758 * exp (1.0219834608 * log ((double) q)));

    for(i = 0; i < fft_count; i++)
    {
      if(multipliers[i] >= estimate)
      {
        *index = i;
        //printf("Index %d\n",*index);
        return  multipliers[i] * 1024;
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
  cudaSetDevice (device_number);
  cutilSafeCall(cudaSetDeviceFlags (cudaDeviceBlockingSync));
  if (d_f)
    {
      cudaGetDeviceProperties (&dev, device_number);
      printf ("------- DEVICE %d -------\n", device_number);
      printf ("name                %s\n", dev.name);
      printf ("Compatibility       %d.%d\n", dev.major, dev.minor);
      printf ("clockRate (MHz)     %d\n", dev.clockRate/1000);
      printf ("memClockRate (MHz)  %d\n", dev.memoryClockRate/1000);
      printf ("totalGlobalMem      %d\n", (int) dev.totalGlobalMem);
      printf ("totalConstMem       %d\n", (int) dev.totalConstMem);
      printf ("l2CacheSize         %d\n", (int) dev.l2CacheSize);
      printf ("sharedMemPerBlock   %d\n", (int) dev.sharedMemPerBlock);
      printf ("regsPerBlock        %d\n", (int) dev.regsPerBlock);
      printf ("warpSize            %d\n", (int) dev.warpSize);
      printf ("memPitch            %d\n", (int) dev.memPitch);
      printf ("maxThreadsPerBlock  %d\n", (int) dev.maxThreadsPerBlock);
      printf ("maxThreadsPerMP     %d\n", dev.maxThreadsPerMultiProcessor);
      printf ("multiProcessorCount %d\n", dev.multiProcessorCount);
      printf ("maxThreadsDim[3]    %d,%d,%d\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
      printf ("maxGridSize[3]      %d,%d,%d\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
      printf ("textureAlignment    %d\n", (int) dev.textureAlignment);
      printf ("deviceOverlap       %d\n\n", dev.deviceOverlap);
      // From Iain
      if (dev.major == 1 && dev.minor < 3)
      {
        printf("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
        exit (2);
      }
      max_threads = (int) dev.maxThreadsPerBlock;
    }
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


unsigned *
read_checkpoint_packed (int q)
{
  //struct stat FileAttrib;
  FILE *fPtr;
  unsigned *x_packed;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;

  x_packed = (unsigned *) malloc (sizeof (unsigned) * (end + 10));
  x_packed[end + 1] = 0;
  x_packed[end + 2] = 1;
  x_packed[end + 3] = (unsigned) -1;
  x_packed[end + 4] = 0;
  if(r_f) return(x_packed);

  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (!fPtr)
  {
//#ifndef _MSC_VER
//    if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the checkpoint file. Trying the backup file.\n");
//#endif
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != (unsigned) q)
  {
    fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return x_packed;
  }
  fPtr = fopen(chkpnt_tfn, "rb");
  if (!fPtr)
  {
//#ifndef _MSC_VER
//    if(stat(chkpnt_cfn, &FileAttrib) == 0) fprintf (stderr, "\nUnable to open the backup file. Restarting test.\n");
//#endif
  }
  else if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
    fclose (fPtr);
  }
  else if(x_packed[end] != (unsigned) q)
  {
    fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
    fclose(fPtr);
  }
  else
  {
    fclose(fPtr);
    return x_packed;
  }
  return x_packed;
}

void pack_bits(double *x, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;

  for(i = 0; i < n; i++)
  {
    temp1 = (int) x[i];
    temp2 += (temp1 << k);
    k += qn + size[i];
    if(k >= 32)
    {
      packed_x[j] = (unsigned) temp2;
      temp2 >>= 32;
      k -= 32;
      j++;
    }
  }
  packed_x[j] = (unsigned) temp2;
}

void set_checkpoint_data(unsigned *x_packed, int q, int n, int j, int offset, int time)
{
  int end = (q + 31) / 32;

  x_packed[end + 5] = q;
  x_packed[end + 6] = n;
  x_packed[end + 7] = j;
  x_packed[end + 8] = offset;
  x_packed[end + 9] = time;
}

void
commit_checkpoint_packed (double *x, unsigned *x_packed, int q, int n)
{
  int i;
  int end = (q + 31) / 32;

  for(i = 0; i < 5; i++) x_packed[end + i] = x_packed[end + i + 5];
  pack_bits(x, x_packed, q, n);

}

void
write_checkpoint_packed (unsigned *x_packed, int q)
{
  FILE *fPtr;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;
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
  fwrite (x_packed, 1, sizeof (unsigned) * (end + 10), fPtr);
  fclose (fPtr);
 if (s_f > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%s", folder, q, x_packed[end + 2], s_residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%s.txt", folder, q, x_packed[end + 2], s_residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr) return;
      fwrite (x_packed, 1, sizeof (unsigned) * (((q + 31) / 32) + 10), fPtr);
      fclose (fPtr);
    }
}




int
printbits (double *x, int q, int n, int offset, FILE* fp, char *expectedResidue, int o_f)
{
  int j, k = 0;
  int digit, bit;
  unsigned long long temp, residue = 0;

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
    sprintf (s_residue, "%016llx", residue);

    printf ("M( %d )C, 0x%s,", q, s_residue);
    if(o_f) printf(" offset = %d,", offset);
    printf (" n = %dK, %s", n/1024, program);
    if (fp)
    {
      fprintf (fp, "M( %d )C, 0x%s,", q, s_residue);
      if(o_f) fprintf(fp, " offset = %d,", offset);
      fprintf (fp, " n = %dK, %s", n/1024, program);
    }
  return 0;
}

void unpack_bits(double *x, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int temp3;
  int i, j = 0, k = 0;
  int qn = q / n;
  int mask1 = -1 << (qn + 1);
  int mask2;
  int mask;

  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  for(i = 0; i < n; i++)
  {
    if(k < qn + size[i])
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(size[i]) mask = mask1;
    else mask = mask2;
    temp3 = ((int) temp2) & mask;
    x[i] = (double) temp3;
    temp2 >>= (qn + size[i]);
    k -= (qn + size[i]);
  }
}

void unpack_bits1(double *x, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int temp3;
  int i, j = 0, k = 0;
  int qn = q / n;
  int mask1 = -1 << (qn + 1);
  int mask2;
  int mask;
  int total = 0;

  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  for(i = 0; i < n; i++)
  {
    if(k < qn + size[i])
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(size[i]) mask = mask1;
    else mask = mask2;
    temp3 = ((int) temp2) & mask;
    if(temp3 != (int) x[i]) total++;
    temp2 >>= (qn + size[i]);
    k -= (qn + size[i]);
  }
  printf("Bad digits: %d\n", total);
}

double* init_lucas_packed(unsigned * x_packed, int q , int *n, int *j, int *offset, int *total_time)
{
  double *x;
  int new_n, old_n;
  int end = (q + 31) / 32;
  int new_test = 0;

  *n = x_packed[end + 1];
  if(*n == 0) new_test = 1;
  *j = x_packed[end + 2];
  *offset = x_packed[end + 3];
  if(total_time) *total_time = x_packed[end + 4];

  old_n = fftlen;
  if(fftlen == 0) fftlen = *n;
  new_n = choose_fft_length(q, &fftlen);
  if(old_n > fft_count) *n = old_n;
  else if (new_test || old_n) *n = new_n;
  init_threads(*n);
  //printf("Using threads: norm1 %d, mult %d, norm2 %d.\n", threads[0], threads[1], threads[2]);
  if ((*n / (2 * threads[0])) > dev.maxGridSize[0])
  {
    fprintf (stderr, "over specifications Grid = %d\n", (int) *n / (2 * threads[0]));
    fprintf (stderr, "try increasing norm1 threads (%d) or decreasing FFT length (%dK)\n\n",  threads[0], *n / 1024);
    return NULL;
  }
  if ((*n / (4 * threads[1])) > dev.maxGridSize[0])
  {
    fprintf (stderr, "over specifications Grid = %d\n", (int) *n / (4 * threads[1]));
    fprintf (stderr, "try increasing mult threads (%d) or decreasing FFT length (%dK)\n\n",  threads[1], *n / 1024);
    return NULL;
  }
  if ((*n % (2 * threads[0]))  != 0)
  {
    fprintf (stderr, "fft length %d must be divisible by 2 * norm1 threads %d\n", *n, threads[0]);
     return NULL;
  }
  if ((*n % (4 * threads[1]))  != 0)
  {
    fprintf (stderr, "fft length %d must be divisible by 4 * mult threads %d\n", *n, threads[1]);
     return NULL;
  }
  if(  *n > 1024 * (2.5 * 0.0000358738168878758 * exp (1.0219834608 * log ((double) q))))
  {
    fprintf (stderr, "The fft length %dK is too large for the exponent %d. Restart with smaller fft.\n", *n / 1024, q);
    return NULL;
  }
  if(  *n < 1024 * ceil(0.95 * 0.0000358738168878758 * exp (1.0219834608 * log ((double) q))))
  {
    fprintf (stderr, "The fft length %dK is too small for the exponent %d. Restart with larger fft.\n", *n / 1024, q);
    return NULL;
  }
  x = (double *) malloc (sizeof (double) * *n);
  get_weights(q, *n);
  alloc_gpu_mem(*n);
  write_gpu_data(q, *n);
  if(!new_test)
  {
    unpack_bits(x, x_packed, q, *n);
    balance_digits(x, q, *n);
  }
  init_x(x, q, *n, offset);
  return x;
}

int isReasonable(int fft)
{ //From an idea of AXN's mentioned on the forums
  int i;

  while(!(fft & 1)) fft /= 2;
  for(i = 3; i <= 7; i += 2) while((fft % i) == 0) fft /= i;
  return (fft);
}

void threadbench (int n, int passes, int device_number)
{
  float total[216] = {0.0f}, outerTime, maxerr = 0.5f;
  int threads[] = {32, 64, 128, 256, 512, 1024};
  int t1, t2, t3, i;
  float best_time = 10000.0f;
  int best_t1 = 0, best_t2 = 0, best_t3 = 0;
  int pass;
  cudaEvent_t start, stop;

  printf("CUDA bench, testing various thread sizes for fft %dK, doing %d passes.\n", n, passes);
  fflush(NULL);
  n *= 1024;

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * n));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * n / 4));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * n / 4));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * n / 32));
  cutilSafeCall (cudaMalloc ((void **) &g_data, sizeof (int) * n / 32));
  cutilSafeCall (cudaMalloc ((void **) &g_carry, sizeof (int) * n / 64));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));

  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
  cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));

  for(t1 = 0; t1 < 6; t1++)
  {
    if(n / (2 * threads[t1]) <= dev.maxGridSize[0] && n % (2 * threads[t1]) == 0)
    {
      for (t2 = 0; t2 < 6; t2++)
      {
        if(n / (4 * threads[t2]) <= dev.maxGridSize[0] && n % (4 * threads[t2]) == 0)
        {
          for (t3 = 0; t3 < 6; t3++)
          {
            for(pass = 1; pass <= passes; pass++)
            {
              cutilSafeCall (cudaEventRecord (start, 0));
              for (i = 0; i < 50; i++)
              {
          cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          rftfsub_kernel <<< n / (4 * threads[t2]), threads[t2] >>> (n, g_x, g_ct);
          cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          norm1 <<<n / (2 * threads[t1]), threads[t1] >>> (g_x, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0);
          norm2 <<< (n / (2 * threads[t1]) + threads[t3] - 1) / threads[t3], threads[t3] >>> (g_x, n, threads[0], g_data, g_carry, g_ttp1);
              }
              cutilSafeCall (cudaEventRecord (stop, 0));
              cutilSafeCall (cudaEventSynchronize (stop));
              cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
              outerTime /= 50.0f;
              total[36 * t1 + 6 * t2 + t3] += outerTime;
              //if(outerTime > max_diff[i]) max_diff[i] = outerTime;
            }
            printf ("fft size = %dK, ave time = %2.4f msec, Norm1 threads %d, Mult threads %d, Norm2 threads %d\n",
                      n / 1024 , total[36 * t1 + 6 * t2 + t3] / passes, threads[t1], threads[t2], threads[t3]);
            fflush(NULL);
          }
        }
      }
    }
  }

  for (i = 0; i < 216; i++)
  {
    if(total[i] < best_time && total[i] > 0.0f)
    {
      int j = i;
      best_time = total[i];
      best_t3 = j % 6;
      j /= 6;
      best_t2 = j % 6;
      best_t1 = j / 6;
    }
  }
  printf("\nBest time for fft = %dK, time: %2.4f, t1 = %d, t2 = %d, t3 = %d\n",
  n/1024, best_time / passes, threads[best_t1], threads[best_t2], threads[best_t3]);

  cufftSafeCall (cufftDestroy (plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_data));
  cutilSafeCall (cudaFree ((char *) g_carry));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));


  char threadfile[32];

  sprintf (threadfile, "%s threads.txt", dev.name);
  FILE *fptr;
  fptr = fopen(threadfile, "a+");
  //if(!fptr) printf("Can't open the damn file.\n");
  if(fptr) fprintf(fptr, "%5d %4d %4d %4d %8.4f\n", n / 1024, threads[best_t1], threads[best_t2], threads[best_t3], best_time / passes);

}


int isprime(unsigned int n)
{
  unsigned int i;

  if(n<=1) return 0;
  if(n>2 && n%2==0)return 0;

  i=3;
  while(i*i <= n && i < 0x10000)
  {
    if(n%i==0)return 0;
    i+=2;
  }
  return 1;
}


void cufftbench (int cufftbench_s, int cufftbench_e, int passes, int device_number)
{
  cudaEvent_t start, stop;
  float outerTime;
  int i, j, k;
  int end = cufftbench_e - cufftbench_s + 1;
  float best_time;
  float *total, *max_diff, maxerr = 0.5f;
  int threads[] = {32, 64, 128, 256, 512, 1024};
  int t1 = 3, t2 = 2, t3 = 2;

  if(end == 1)
  {
    threadbench(cufftbench_e, passes, device_number);
    return;
  }

  printf ("CUDA bench, testing reasonable fft sizes %dK to %dK, doing %d passes.\n", cufftbench_s, cufftbench_e, passes);

  total = (float *) malloc (sizeof (float) * end);
  max_diff = (float *) malloc (sizeof (float) * end);
  for(i = 0; i < end; i++)
  {
    total[i] = max_diff[i] = 0.0f;
  }

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_ttmp, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * 1024 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_ct, sizeof (double) * 256 * cufftbench_e));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * 256 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_ttp1, sizeof (double) * 1024 / 32 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_data, sizeof (int) * 1024 / 32 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_carry, sizeof (int) * 512 / 32 * cufftbench_e));
  cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));

  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));

  for (j = cufftbench_s; j <= cufftbench_e; j++)
  {
    if(isReasonable(j) < 15)
    {
      int n = j * 1024;
      cufftSafeCall (cufftPlan1d (&plan, n / 2, CUFFT_Z2Z, 1));
      for(k = 0; k < passes; k++)
      {
        cutilSafeCall (cudaEventRecord (start, 0));
        for (i = 0; i < 50; i++)
  	    {
          cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          rftfsub_kernel <<< n / (4 * threads[t2]), threads[t2] >>> (n, g_x, g_ct);
          cufftSafeCall (cufftExecZ2Z (plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          norm1 <<<n / (2 * threads[t1]), threads[t1] >>> (g_x, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0);
          norm2 <<< (n / (2 * threads[t1]) + threads[t3] - 1) / threads[t3], threads[t3] >>> (g_x, n, threads[0], g_data, g_carry, g_ttp1);
        }
        cutilSafeCall (cudaEventRecord (stop, 0));
        cutilSafeCall (cudaEventSynchronize (stop));
        cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
        i = j - cufftbench_s;
        outerTime /= 50.0f;
        total[i] += outerTime;
        if(outerTime > max_diff[i]) max_diff[i] = outerTime;
      }
      cufftSafeCall (cufftDestroy (plan));
      printf ("fft size = %dK, ave time = %2.4f msec, max-ave = %0.5f\n",
                  j , total[i] / passes, max_diff[i] - total[i] / passes);
      fflush(NULL);
    }
  }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_ttmp));
  cutilSafeCall (cudaFree ((char *) g_ttp1));
  cutilSafeCall (cudaFree ((char *) g_ct));
  cutilSafeCall (cudaFree ((char *) g_data));
  cutilSafeCall (cudaFree ((char *) g_carry));
  cutilSafeCall (cudaFree ((char *) g_err));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));

  i = end - 1;
  best_time = 10000.0f;
  while(i >= 0)
  {
    if(total[i] > 0.0f && total[i] < best_time) best_time = total[i];
    else total[i] = 0.0f;
    i--;
  }
  char fftfile[32];
  FILE *fptr;

  sprintf (fftfile, "%s fft.txt", dev.name);
  fptr = fopen(fftfile, "w");
  if(!fptr)
  {
    printf("Cannot open %s.\n",fftfile);
    printf ("Device              %s\n", dev.name);
    printf ("Compatibility       %d.%d\n", dev.major, dev.minor);
    printf ("clockRate (MHz)     %d\n", dev.clockRate/1000);
    printf ("memClockRate (MHz)  %d\n", dev.memoryClockRate/1000);
    printf("\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
         int tl = (int) (exp(0.9784876919 * log (cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        printf("%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fflush(NULL);
  }
  else
  {
    fprintf (fptr, "Device              %s\n", dev.name);
    fprintf (fptr, "Compatibility       %d.%d\n", dev.major, dev.minor);
    fprintf (fptr, "clockRate (MHz)     %d\n", dev.clockRate/1000);
    fprintf (fptr, "memClockRate (MHz)  %d\n", dev.memoryClockRate/1000);
    fprintf(fptr, "\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
        int tl = (int) (exp(0.9784876919 * log (cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        //fprintf(fptr, "%5d", (cufftbench_s + i)/2);
        //if((cufftbench_s + i) %2) fprintf(fptr, ".5");

        fprintf(fptr, "%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fclose(fptr);
    printf("Optimal fft lengths saved in %s.\nPlease email a copy to james@mersenne.ca.\n", fftfile);
    fflush(NULL);
   }

  free ((char *) total);
  free ((char *) max_diff);
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

int round_off_test(double *x, int q, int n, int *j, int *offset)
{
  int k;
  float totalerr = 0.0;
  float terr, avgerr, maxerr = 0.0;
  float max_err = 0.0, max_err1 = 0.0;
  int l_offset = *offset;
  int last = q - 2;

      printf("Running careful round off test for 1000 iterations.\n");
      printf("If average error > 0.25, or maximum error > 0.35,\n");
      printf("the test will restart with a longer FFT.\n");
      fflush(NULL);
      for (k = 0; k < 1000 && (k + *j <= last); k++)
	    {
        terr = lucas_square (x, q, n, *j + k, q - 1, &l_offset, &maxerr, 1);
        if(terr > maxerr) maxerr = terr;
        if(terr > max_err) max_err = terr;
        if(terr > max_err1) max_err1 = terr;
        totalerr += terr;
        reset_err(&maxerr, 0.0);
        if(terr > 0.35)
        {
	        printf ("Iteration = %d < 1000 && err = %5.5f > 0.35, increasing n from %dK\n", k, terr, n/1024);
	        fftlen++;
	        return 1;
        }
	      if( k && (k % 100 == 0) )
        {
	        printf( "Iteration  %d, average error = %5.5f, max error = %5.5f\n", k, totalerr / k, max_err);
	        max_err = 0.0;
	      }
	    }
      avgerr = totalerr/1000.0;
      if( avgerr > 0.25 )
      {
        printf("Iteration 1000, average error = %5.5f > 0.25 (max error = %5.5f), increasing FFT length and restarting\n", avgerr, max_err);
        fftlen++;
        return 1;
      }
      else if( avgerr < 0 )
      {
        fprintf(stderr, "Something's gone terribly wrong! Avgerr = %5.5f < 0 !\n", avgerr);
        exit (2);
      }
      else
      {
        printf("Iteration 1000, average error = %5.5f <= 0.25 (max error = %5.5f), continuing test.\n", avgerr, max_err1);
        //reset_err(&maxerr, 0.0);
      }
      *offset = l_offset;
      *j += 1000;
      return 0;
}

int interact(int );
/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int
check (int q, char *expectedResidue)
{
  int n, j, last = q - 2, error_flag;
  double  *x = NULL;
  unsigned *x_packed = NULL;
  float maxerr, terr;
  int restarting = 0;
  timeval time0, time1;
  int j_save = 0, offset_save = 0;
  int total_time = 0, start_time;
   // use start_total because every time we write a checkpoint, total_time is increased,
   // but we don't quit everytime we write a checkpoint
  int offset;
  int reset_fft = 0;
  int j_resume = 0, last_chk = 0;
  int interact_result = 0;

  signal (SIGTERM, SetQuitting);
  signal (SIGINT, SetQuitting);
  do
  {				/* while (restarting) */
    maxerr = 0.0;

    if(!x_packed) x_packed = read_checkpoint_packed(q);
    x = init_lucas_packed (x_packed, q, &n, &j, &offset, &total_time);
    if(!x) exit (2);


    restarting = 0;
    if(j == 1)
    {
      if(!restarting) printf ("Starting M%d fft length = %dK\n", q, n/1024);
      restarting = round_off_test(x, q, n, &j, &offset);
      if(!restarting)
	    {
	      cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
        standardize_digits(x, q, n, 0, n);
        set_checkpoint_data(x_packed, q, n, j, offset, total_time);
	      commit_checkpoint_packed (x, x_packed, q, n);
        write_checkpoint_packed (x_packed, q);
	      if(checkpoint_iter > 1000) j_resume = 1001;
	      last_chk = j_resume / checkpoint_iter;
      }
    }
    else
    {
      printf ("Continuing work from a partial result of M%d fft length = %dK iteration = %d\n", q, n/1024, j);
      j_resume = j % checkpoint_iter - 1;
      last_chk = j / checkpoint_iter;
    }
    fflush (stdout);
    if(!restarting)
    {
      gettimeofday (&time0, NULL);
      start_time = time0.tv_sec;
    }

    for (; !restarting && j <= last; j++) // Main LL loop
    {
	    if ((j % 100) == 0) error_flag = 1;
	    else error_flag = 0;
      terr = lucas_square (x, q, n, j, last, &offset, &maxerr, error_flag);
	    if ((j % checkpoint_iter) == 0 || quitting)
      {
	      cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
        standardize_digits(x, q, n, 0, n);
	      gettimeofday (&time1, NULL);
        total_time += (time1.tv_sec - start_time);
        start_time = time1.tv_sec;
        set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	      if(quitting)
	      {
          cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
	        if(terr <= 0.40)
	        {
	          commit_checkpoint_packed (x, x_packed, q, n);
	          write_checkpoint_packed (x_packed, q);
	        }
	        printf(" Estimated time spent so far: ");
	        print_time_from_seconds(total_time);
	        printf("\n\n");
		      j = last + 1;
	      }
	      else
        {
          printf ("Iteration %d ", j);
          printbits (x, q, n, offset, 0, expectedResidue, 0);
	        long long diff = time1.tv_sec - time0.tv_sec;
	        long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
	        long long diff2 = (last - j) * diff1 / (((j / checkpoint_iter - last_chk) * checkpoint_iter - j_resume) *  1e6);
	        gettimeofday (&time0, NULL);
	        printf (" err = %5.5f (", maxerr);
	        print_time_from_seconds ((int) diff);
	        printf (" real, %4.4f ms/iter, ETA ", diff1 / 1000.0 / ((j / checkpoint_iter - last_chk) * checkpoint_iter - j_resume));
	        print_time_from_seconds ((int) diff2);
	        printf (")\n");
	        fflush (stdout);
	        if(j_resume) j_resume = 0;
	        last_chk = j /checkpoint_iter;
	        reset_err(&maxerr, error_reset / (float) 100.0); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
	      }
	    }
      if (error_flag)
	    {
	      int end = (q + 31) / 32;
	      if (terr > 0.40)
		    {
		      if (t_f)
	        {
		        gettimeofday(&time1, NULL);
		        printf ("Iteration = %d >= 1000 && err = %5.5f > 0.40, fft length = %dK, writing checkpoint file (because -t is enabled) and exiting.\n\n", j, terr, n/1024);
		        cutilSafeCall (cudaMemcpy (x, g_save, sizeof (double) * n, cudaMemcpyDeviceToHost));
		        total_time += (time1.tv_sec - start_time);
	          set_checkpoint_data(x_packed, q, n, j_save + 1, offset_save, total_time);
	          commit_checkpoint_packed (x, x_packed, q, n);
	          write_checkpoint_packed (x_packed, q);
		      }
		      printf ("Iteration = %d >= 1000 && err = %5.5g > 0.40, fft length = %dK, restarting from last checkpoint with longer fft.\n\n", j, (double) terr, (int) n / 1024);
          fftlen++;
          restarting = 1;
          reset_fft = 1;
          reset_err(&maxerr, 0.0);
	      }
	      else		// error_flag && terr <= 0.40
	      {
          if(x_packed[end + 3] != x_packed[end + 8])
          {
            commit_checkpoint_packed (x, x_packed, q, n);
            write_checkpoint_packed (x_packed, q);
            if(reset_fft)
            {
              printf("Sticking point passed. Switching back to shorter fft.\n");
              fftlen--;
              restarting = 1;
              reset_fft = 0;
              reset_err(&maxerr, 0.0);
            }
          }
		      if (t_f)
		      {
		        copy_kernel <<< n / 128, 128 >>> (g_save, g_x);
		        j_save = j;
		        offset_save = offset;
		      }
		    }
	    }

      if ( k_f && !quitting && (!(j & 15)) && _kbhit()) interact_result = interact(n); // abstracted to clean up check()
      if(interact_result == 2) reset_fft = 0;
      if(interact_result == 1)
      {
        if((j % checkpoint_iter) != 0)
        {
           cutilSafeCall (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
           if(terr <= 0.40)
           {
		         cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
             standardize_digits(x, q, n, 0, n);
	           set_checkpoint_data(x_packed, q, n, j + 1, offset, total_time);
	           commit_checkpoint_packed (x, x_packed, q, n);
             reset_err(&maxerr, 0.0);
           }
        }
        restarting = 1;
      }
	    interact_result = 0;
	    fflush (stdout);
	  } /* end main LL for-loop */

    if (!restarting && !quitting)
	  { // done with test
	    gettimeofday (&time1, NULL);
	    FILE* fp = fopen_and_lock(RESULTSFILE, "a");
	    if(!fp)
	    {
	      fprintf (stderr, "Cannot write results to %s\n\n", RESULTSFILE);
	      exit (1);
	    }
	    cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
      if (standardize_digits(x, q, n, 0, n))
      {
        printf ("M( %d )P, n = %dK, %s", q, n / 1024, program);
        if (fp) fprintf (fp, "M( %d )P, n = %dK, %s", q, n / 1024, program);
      }
	    else printbits (x, q, n, offset, fp, 0, 1);
      total_time += (time1.tv_sec - start_time);
      printf (", estimated total time = ");
      print_time_from_seconds(total_time);

	    if( AID[0] && strncasecmp(AID, "N/A", 3) )
      { // If (AID is not empty), AND (AID is NOT "N/A") (case insensitive)
        fprintf(fp, ", AID: %s\n", AID);
	    }
      else fprintf(fp, "\n");
	    unlock_and_fclose(fp);
	    fflush (stdout);
	    rm_checkpoint (q);
	    printf("\n\n");
	  }
    close_lucas (x);
  }
  while (restarting);
  free ((char *) x_packed);
  return (0);
}

int
check_residue (int q, char *expectedResidue)
{
  int n, j, last, offset;
  unsigned *x_packed = NULL;
  double  *x = NULL;
  float maxerr = 0.0;
  int restarting = 0;
  timeval time0, time1;

  do
  {
    if(!x_packed) x_packed = read_checkpoint_packed(q);
    x = init_lucas_packed (x_packed, q, &n, &j, &offset, NULL);
    if(!x) exit (2);
    gettimeofday (&time0, NULL);
    if(!restarting) printf ("Starting self test M%d fft length = %dK\n", q, n/1024);
    restarting = round_off_test(x, q, n, &j, &offset);
    if(restarting) close_lucas (x);
  }
  while (restarting);

  fflush (stdout);
  last = 10000;

  for (; j <= last; j++)
  {
    lucas_square (x, q, n, j, last, &offset, &maxerr, 0);
    if(j % 100 == 0) cutilSafeCall (cudaMemcpy (&maxerr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
  }
  cutilSafeCall (cudaMemcpy (x, g_x, sizeof (double) * n, cudaMemcpyDeviceToHost));
  standardize_digits(x, q, n, 0, n);
  gettimeofday (&time1, NULL);

  printf ("Iteration %d ", j - 1);
  printbits (x, q, n, offset, 0, expectedResidue, 0);
  long long diff = time1.tv_sec - time0.tv_sec;
  long long diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
  printf (" err = %5.5f (", maxerr);
  print_time_from_seconds ((int) diff);
  printf (" real, %4.4f ms/iter", diff1 / (1000.0 * last));
  printf (")\n");

  fftlen = 0;
  close_lucas (x);
  free ((char *) x_packed);
  if (strcmp (s_residue, expectedResidue))
  {
    printf("Expected residue [%s] does not match actual residue [%s]\n\n", expectedResidue, s_residue);
    fflush (stdout);
    return 1;
  }
  else
  {
    printf("This residue is correct.\n\n");
    fflush (stdout);
    return (0);
  }
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
  threads[0] = -1;
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
   if( threads[0] < 1 && 			!IniGetInt3(INIFILE, "Threads", &threads[0], &threads[1], &threads[2], THREADS_DFLT) )
    fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: %d\n", THREADS_DFLT);
   if( s_f < 0 && 			!IniGetInt(INIFILE, "SaveAllCheckpoints", &s_f, S_F_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
   if( 		     	     s_f > 0 && !IniGetStr(INIFILE, "SaveFolder", folder, SAVE_FOLDER_DFLT) )
    /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
   /*if( t_f < 0 && 			!IniGetInt(INIFILE, "CheckRoundoffAllIterations", &t_f, 0) )
    fprintf(stderr, "Warning: Couldn't parse ini file option CheckRoundoffAllIterations; using default: off\n");*/
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
      if( threads[0] < 1 ) threads[0] = THREADS_DFLT;
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
  if (threads[0] != 32 && threads[0] != 64 && threads[0] != 128
	      && threads[0] != 256 && threads[0] != 512 && threads[0] != 1024)
  {
    fprintf(stderr, "Error: thread count is invalid.\n");
    fprintf(stderr, "Threads must be 2^k, 5 <= k <= 10.\n\n");
    exit(2);
  }
  f_f = fftlen; // if the user has given an override... then note this length must be kept between tests


  init_device (device_number);
  fft_count = init_ffts();

  if (r_f)
    {
      int bad_selftest = 0;
      fftlen = 0;
      bad_selftest += check_residue (86243, "23992ccd735a03d9");
      bad_selftest += check_residue (132049, "4c52a92b54635f9e");
      bad_selftest += check_residue (216091, "30247786758b8792");
      bad_selftest += check_residue (756839, "5d2cbe7cb24a109a");
      bad_selftest += check_residue (859433, "3c4ad525c2d0aed0");
      bad_selftest += check_residue (1257787, "3f45bf9bea7213ea");
      bad_selftest += check_residue (1398269, "a4a6d2f0e34629db");
      bad_selftest += check_residue (2976221, "2a7111b7f70fea2f");
      bad_selftest += check_residue (3021377, "6387a70a85d46baf");
      bad_selftest += check_residue (6972593, "88f1d2640adb89e1");
      bad_selftest += check_residue (13466917, "9fdc1f4092b15d69");
      bad_selftest += check_residue (20996011, "5fc58920a821da11");
      bad_selftest += check_residue (24036583, "cbdef38a0bdc4f00");
      bad_selftest += check_residue (25964951, "62eb3ff0a5f6237c");
      bad_selftest += check_residue (30402457, "0b8600ef47e69d27");
      bad_selftest += check_residue (32582657, "02751b7fcec76bb1");
      bad_selftest += check_residue (37156667, "67ad7646a1fad514");
      bad_selftest += check_residue (42643801, "8f90d78d5007bba7");
      bad_selftest += check_residue (43112609, "e86891ebf6cd70c4");
      bad_selftest += check_residue (57885161, "76c27556683cd84d");
      if (bad_selftest)
      {
        fprintf(stderr, "Error: There ");
        bad_selftest > 1 ? fprintf(stderr, "were %d bad selftests!\n",bad_selftest)
        		 : fprintf(stderr, "was a bad selftest!\n");
      }
    }
  else if (cufftbench_d > 0)
  {
    cufftbench (cufftbench_s, cufftbench_e, cufftbench_d, device_number);
  }
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
	       "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-polite iteration] [-k] exponent|input_filename\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
      	  fprintf (stderr,
	       "$ CUDALucas [-d device_number] [-info] -cufftbench start end passes\n\n");
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
	       "                       -polite     GPU is polite every n iterations (default -polite 1) (-polite 0 = GPU aggressive)\n");
      	  fprintf (stderr,
	       "                       -cufftbench exec CUFFT benchmark. Example: $ ./CUDALucas -d 1 -cufftbench 1 8192 2\n");
      	  fprintf (stderr,
	       "                        checks iteration times for reasonable fft lengths between 1K and 8192k averaging over 2 passes\n");
      	  fprintf (stderr,
      	 "                       -r          exec residue test.\n");
      	  fprintf (stderr,
	       "                       -k          enable keys (see CUDALucas.ini for details.)\n\n");
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
	  threads[0] = atoi (argv[2]);
	  if (threads[0] != 32 && threads[0] != 64 && threads[0] != 128
	      && threads[0] != 256 && threads[0] != 512 && threads[0] != 1024)
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
	  } else {
    *q = derp;
    *q |= 1;
    while(!isprime(*q)) *q += 2;
}
	  argv++;
	  argc--;
	}
    }
}

int interact(int n)
{
  int c = getchar ();
  int k, l;

     switch( c )
     {
        case 'p' :
                    polite_f = 1 - polite_f;
                    printf ("   -polite %d\n", polite_f * polite);
                    break;
       //case 't' :
       //             t_f = 0;
       //             printf ("   disabling -t\n");
       //             break;
        case 's' :
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
                    break;
        case 'F' :
                    printf(" -- Increasing fft length.\n");
                    fftlen++;
                    return 1;
        case 'f' :
                    printf(" -- Decreasing fft length.\n");
                    fftlen--;
                    return 1;
        case 'c' :
                    printf(" -- fft length reset cancelled.\n");
                    return 2;
        case 'Q' :
                    cutilSafeThreadSync();
                    if(threads[0] < max_threads) threads[0] *= 2;
                    l = 0;
                    k = 32;
                    while(k < threads[0])
                    {
                      l += n / (2 * k);
                      k *= 2;
                    }
                    g_ttp1 = &g_ttp2[l];
                    printf(" -- threads increased to %d.\n", threads[0]);
                    break;
        case 'q' :
                    cutilSafeThreadSync();
                    if(threads[0] > 32 && (n / threads[0]) <= 65535) threads[0] /= 2;
                    l = 0;
                    k = 32;
                    while(k < threads[0])
                    {
                      l += n / (2 * k);
                      k *= 2;
                    }
                    g_ttp1 = &g_ttp2[l];
                    printf(" -- threads decreased to %d.\n", threads[0]);
                    break;
        case 'W' :
                    if(threads[1] < 1024) threads[1] *= 2;
                    printf(" -- threads1 increased to %d.\n", threads[1]);
                    break;
        case 'w' :
                    if(threads[1] > 32) threads[1] /= 2;
                    printf(" -- threads1 decreased to %d.\n", threads[1]);
                    break;
        case 'E' :
                    if(threads[2] < max_threads) threads[2] *= 2;
                    printf(" -- threads2 increased to %d.\n", threads[2]);
                    break;
        case 'e' :
                    if(threads[2] > 32) threads[2] /= 2;
                    printf(" -- threads2 decreased to %d.\n", threads[2]);
                    break;
        case 'R' :
                    if(error_reset < 100) error_reset += 5;
                    printf(" -- error_reset increased to %d.\n", error_reset);
                    break;
        case 'r' :
                    if(error_reset > 0) error_reset -= 5;
                    printf(" -- error_reset decreased to %d.\n", error_reset);
                    break;
        case 'T' :
                    if(checkpoint_iter == 1) checkpoint_iter = 2;
                    else if(checkpoint_iter == 2) checkpoint_iter = 5;
                    else
                    {
                      k = checkpoint_iter;
                      while(k % 10 == 0) k /= 10;
                      if(k == 1) checkpoint_iter *= 2.5;
                      else checkpoint_iter *= 2;
                    }
                    printf(" -- checkpoint_iter increased to %d.\n", checkpoint_iter);
                    break;
        case 't' :
                    k = checkpoint_iter;
                    if(checkpoint_iter == 5) checkpoint_iter = 2;
                    else
                    {
                      while (k % 10 == 0) k /= 10;
                      if (k == 25) checkpoint_iter /= 2.5;
                      else if (checkpoint_iter > 1) checkpoint_iter /= 2;
                    }
                    printf(" -- checkpoint_iter decreased to %d.\n", checkpoint_iter);
                    break;
        default  :
                    break;
     }
   fflush (stdin);
   return 0;
}
