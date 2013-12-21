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
//#include "nvml.h"
#include "cuda_safecalls.h"
#include "parse.h"


/* In order to have the gettimeofday() function, you need these includes on Linux:
#include <sys/time.h>
#include <unistd.h>
On Windows, you need
#include <winsock2.h> and a definition for
int gettimeofday (struct timeval *tv, struct timezone *) {}
Both platforms are taken care of in parse.h and parse.c. */
/* http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html
   base code from Takuya OOURA.  */

/************************ definitions ************************************/
#ifdef _MSC_VER
#define strncasecmp strnicmp
#endif

/**************************************************************
***************************************************************
*                                                             *
*                       Global Variables                      *
*                                                             *
***************************************************************
**************************************************************/
double *g_x;                      //gpu data
double *g_ttmp;                   //weighting factors
double *g_ttp1;                   //weighting factors for splicing step
double *g_ct;                     //factors used in multiplication kernel enabling real as complex
int    *g_data;                   //integer values of data for splicing step
int    *g_carry;                  //carry data for splicing step
int    *g_xint;                   //integer copy of gpu data for transfer to cpu
char   *g_size;                   //information on the number of bits each digit of g_x uses, 0 or 1
float  *g_err;                    //current maximum error

__constant__ double g_ttpinc[2];  //factors for obtaining weights of adjacent digits
__constant__ int    g_qn[1];      //base size of bit values for each digit, adjusted by size data above

cufftHandle    g_plan;
cudaDeviceProp g_dev;             //structure holding device property information

int g_ffts[256];                  //array of ffts lengths
int g_fft_count = 0;              //number of ffts lengths in g_ffts
int g_fftlen;                     //index of the fft currently in use
int g_lbi = 0;                    //index of the smallest feasible fft length for the current exponent
int g_ubi = 256;                  //index of the largest feasible fft length for the current exponent
int g_thr[3] = {256, 128, 128};   //current threads values

int g_cpi;                        //checkpoint interval
int g_er = 85;                    //error reset value
int g_po;                         //polite value

int g_qu;                         //quitting flag
int g_sf;                         //safe all checkpoint files flag
int g_rt;                         //residue check flag
int g_ro;                         //roundoff test flag
int g_dn;                         //device number
int g_df;                         //show device info flag
int g_ki;                         //keyboard input flag
int g_pf;                         //polite flag
int g_th;                         //threads flag

char g_folder[192];               //folder where savefiles will be kept
char g_input_file[192];           //input file name default worktodo.txt
char g_RESULTSFILE[192];          //output file name default results.txt
char g_INIFILE[192] = "CUDALucas.ini"; //initialization file name
char g_AID[192];                  // Assignment key
char g_output[50][32];
char g_output_string[192];
char g_output_header[192];
int  g_output_code[50];
int  g_output_interval;
/**************************************************************
***************************************************************
*                                                             *
*             Kernels and other device functions              *
*                                                             *
***************************************************************
**************************************************************/


/**************************************************************
*                                                             *
*             Functions for setting constant memory           *
*                                                             *
**************************************************************/

// Factors to convert from one weighting factor to an adjacent
// weighting factor.
void set_ttpinc(double *h_data)
{
  cudaMemcpyToSymbol(g_ttpinc, h_data, 2 * sizeof(double));
}

// Set to q/n, the number of bits each digit uses is given by
// numbits in the rcb and splicing kernels. Numbits is q/n + size.
// We can avoid many memory transfers by having this in constant
// memory.
void set_qn(int *h_qn)
{
  cudaMemcpyToSymbol(g_qn, h_qn, 1 * sizeof(int));
}

/**************************************************************
*                                                             *
*                       Device functions                      *
*                                                             *
**************************************************************/

// inline ptx rounding function
# define RINT(x)  __rintd(x)

__device__ static double __rintd (double z)
{
  double y;
  asm ("cvt.rni.f64.f64 %0, %1;": "=d" (y):"d" (z));
  return (y);
}

/**************************************************************
*                                                             *
*                           Kernels                           *
*                                                             *
**************************************************************/

// These two used in memtest only
__global__ void copy_kernel (double *in, int n, int pos, int s)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;

    in[s * n + index] = in[pos * n + index];
}


__global__ void compare_kernel (double *in1,  double *in2, volatile int *compare)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  int temp;
  double d1, d2;

  d1 = in1[threadID];
  d2 = in2[threadID];
  temp = (d1 != d2);
  if(temp > 0) atomicAdd((int *) compare, 1);
}

// Applies the irrational base weights to balanced integer data.
// Used in initialization at the (re)commencement of a test.
__global__ void apply_weights (double *out, int *in, double *ttmp)
{
  int val[2], test = 1;
  double ttp_temp[2];
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;

  val[0] = in[index];
  val[1] = in[index + 1];
  ttp_temp[0] = ttmp[index];
  ttp_temp[1] = ttmp[index + 1];
  if(ttp_temp[0] < 0.0) test = 0;
  if(ttp_temp[1] < 0.0) ttp_temp[1] = -ttp_temp[1];
  out[index + 1] = (double) val[1] * ttp_temp[1];
  ttp_temp[1] *= -g_ttpinc[test];
  out[index] = (double) val[0] * ttp_temp[1];
}

// The pointwise multiplication of the fft'ed data.
// We are using real data interpreted as complex with odd
// indices being the imaginary part. The work with the number
// (wkr, wki) is needed for this.
__global__ void square (int n, double *a, double *ct)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int nc = n >> 2;
  double wkr, wki, xr, xi, yr, aj, aj1, ak, ak1;

  if (j)
  {
    wkr = 0.5 - ct[nc - j];
    wki = ct[j];
    j <<= 1;
    nc = n - j;
    aj = a[j];
    aj1 = a[1 + j];
    ak = a[nc];
    ak1 = a[1 + nc];

    xr = aj - ak;
    xi = aj1 + ak1;
    yr = wkr * xr - wki * xi;
    xi = wkr * xi + wki * xr;

    aj -= yr;
    aj1 -= xi;
    ak += yr;
    ak1 -= xi;

    xr = (aj - aj1) * (aj + aj1);
    xi = 2.0 * aj * aj1;
    yr = (ak - ak1) * (ak + ak1);
    ak = 2.0 * ak * ak1;

    aj1 = xr - yr;
    ak1 = xi + ak;
    aj = wkr * aj1 + wki * ak1;
    aj1 = wkr * ak1 - wki * aj1;

    a[j] = xr - aj;
    a[1 + j] = aj1 - xi;
    a[nc] = yr + aj;
    a[1 + nc] =  aj1 - ak;
  }
  else
  {
    j = n >> 1;
    aj = a[0];
    aj1 = a[1];
    xi = aj - aj1;
    aj += aj1;
    xi *= xi;
    aj *= aj;
    aj1 = 0.5 * (xi - aj);
    a[0] = aj + aj1;
    a[1] = aj1;
    xr = a[0 + j];
    xi = a[1 + j];
    a[1 + j] = -2.0 * xr * xi;
    a[0 + j] = (xr + xi) * (xr - xi);
  }
}


// Rounding, error checking, carrying, balancing kernel.
// This version uses 32 bit carries which a vast majority of
// the time is sufficient. The check for overflow increases
// iteration times by about 0.1%.
__global__ void rcb1 (double *in,
                         int *xint,
                         int *data,
                         double *ttmp,
                         int *carry_out,
		                     volatile float *err,
		                     float maxerr,
		                     int digit,
		                     int bit,
		                     int cp_flag)
{
  long long int bigint[2];
  int numbits[2] = {g_qn[0],g_qn[0]};
  double ttp_temp;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  const int index1 = blockIdx.x << 1;
  __shared__ int carry[1024 + 1];

  {
    double tval[2], trint[2];
    float ferr[2];

    // Processing two per thread to save on memory traffic.
    tval[0] = ttmp[index];
    ttp_temp = ttmp[index + 1];
    trint[0] = in[index];
    trint[1] = in[index + 1];

    // Size is coded into the sign of the weighting factors,
    // Adjust numbits and weighting factors accordingly
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
    //unweight, round, and check errors
    tval[1] = trint[1] * tval[0] * g_ttpinc[numbits[0] == g_qn[0]];
    tval[0] = trint[0] * tval[0];
    trint[0] = RINT (tval[0]);
    trint[1] = RINT (tval[1]);
    bigint[0] = (long long int) trint[0];
    bigint[1] = (long long int) trint[1];
    ferr[0] = fabs (tval[0] - trint[0]);
    ferr[1] = fabs (tval[1] - trint[1]);
    if(ferr[0] < ferr[1]) ferr[0] = ferr[1];
    if(ferr[0] > maxerr) atomicMax((int*) err, __float_as_int(ferr[0]));
  }
  {
    int val[2], mask[2], shifted_carry;

    mask[0] = -1 << numbits[0];
    mask[1] = -1 << numbits[1];

    // subtact 2 from the appropriate position
    if(index == digit) bigint[0] -= bit;
    else if((index + 1) == digit) bigint[1] -= bit;

    // Compute 1st carry and check for overflow
    val[1] = ((int) bigint[1]) & ~mask[1];
    bigint[1] >>= numbits[1];
    if( abs(bigint[1]) > 0x000000007fffffff) atomicMax((int*) err, __float_as_int(1.0f));
    val[0] = ((int) bigint[0]) & ~mask[0];
    bigint[0] >>= numbits[0];
    if( abs(bigint[0]) > 0x000000007fffffff) atomicMax((int*) err, __float_as_int(1.0f));
    carry[threadIdx.x + 1] = (int) (bigint[1]);
    val[1] += (int) (bigint[0]);
    __syncthreads ();

    // Add 1st carry and compute second carry
    if (threadIdx.x) val[0] += carry[threadIdx.x];
    shifted_carry = val[1] - (mask[1] >> 1);
    val[1] -= shifted_carry & mask[1];
    carry[threadIdx.x] = shifted_carry >> numbits[1];
    shifted_carry = val[0] - (mask[0] >> 1);
    // Add second carry
    val[0] -= shifted_carry & mask[0];
    val[1] += shifted_carry >> numbits[0];
    __syncthreads ();

    // write end of block carry to global memory for the splicing kernel
    if (threadIdx.x == (blockDim.x - 1))
    {
      if (blockIdx.x == gridDim.x - 1) carry_out[0] = carry[threadIdx.x + 1] + carry[threadIdx.x];
      else   carry_out[blockIdx.x + 1] =  carry[threadIdx.x + 1] + carry[threadIdx.x];
    }

    if (threadIdx.x) // these indices don't need the splicing kernel
    {
      //apply weights before writing back to memory
      val[0] += carry[threadIdx.x - 1];
      {
        in[index + 1] = (double) val[1] * ttp_temp;
        ttp_temp *= -g_ttpinc[numbits[0] == g_qn[0]];
        in[index] = (double) val[0] * ttp_temp;
      }
      // write integer data to xint for transfer to host when savefiles need writing
      if(cp_flag)
      {
        xint[index + 1] = val[1];
        xint[index] = val[0];
      }
    }
    // to be sent to the splicing kernel, leave as integer data
    else
    {
      data[index1] = val[0];
      data[index1 + 1] = val[1];
    }
  }
}

// Splicing kernel, 32 bit version. Handles carries between blocks of rcb1
__global__ void splice1 (double *out,
                         int *xint,
                         int n,
                         int threads,
                         int *data,
                         int *carry_in,
                         double *ttp1,
                         int cp_flag)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads * threadID) << 1;
  int temp0, temp1;
  int mask, shifted_carry, numbits= g_qn[0];
  double temp;

  if (j < n) //make sure this index actually refers to some data
  {
    temp0 = data[threadID1] + carry_in[threadID];
    temp1 = data[threadID1 + 1];
    temp = ttp1[threadID];
    //extract and apply size information
    if(temp < 0.0)
    {
      numbits++;
      temp = -temp;
    }
    //do the last carry
    mask = -1 << numbits;
    shifted_carry = temp0 - (mask >> 1) ;
    temp0 = temp0 - (shifted_carry & mask);
    temp1 += (shifted_carry >> numbits);
    //apply weights before writing to memory
    out[j + 1] = temp1 * temp;
    temp *= -g_ttpinc[numbits == g_qn[0]];
    out[j] = temp0 * temp;
    //write data to integer array when savefiles need writing
    if(cp_flag)
    {
      xint[j + 1] = temp1;
      xint[j] = temp0;
    }
  }
}

// Rounding, error checking, slower 64 bit version for fixing overflow errors.
__global__ void rcb2 (double *in,
                         int *xint,
                         long long int *data,
                         double *ttmp,
                         long long int *carry_out,
		                     volatile float *err,
		                     float maxerr,
		                     int digit,
		                     int bit,
		                     int err_flag)
{
  long long int bigint[2];
  int  numbits[2] = {g_qn[0],g_qn[0]};
  double ttp_temp;
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) << 1;
  const int index1 = blockIdx.x << 1;
  __shared__ long long int carry[1024 + 1];

  {
    double tval[2], trint[2];
    float ferr[2];

    tval[0] = ttmp[index];
    ttp_temp = ttmp[index + 1];
    trint[0] = in[index];
    trint[1] = in[index + 1];
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
    tval[1] = trint[1] * tval[0] * g_ttpinc[numbits[0] == g_qn[0]];
    tval[0] = trint[0] * tval[0];
    trint[0] = RINT (tval[0]);
    trint[1] = RINT (tval[1]);
    bigint[0] = (long long int) trint[0];
    bigint[1] = (long long int) trint[1];
    ferr[0] = fabs (tval[0] - trint[0]);
    ferr[1] = fabs (tval[1] - trint[1]);
    if(ferr[0] < ferr[1]) ferr[0] = ferr[1];
    if(ferr[0] > maxerr) atomicMax((int*) err, __float_as_int(ferr[0]));
  }
  {
    long long int shifted_carry;
    int  mask[2];

    mask[0] = -1 << numbits[0];
    mask[1] = -1 << numbits[1];
    if(index == digit) bigint[0] -= bit;
    else if((index + 1) == digit) bigint[1] -= bit;
    carry[threadIdx.x + 1] = (bigint[1] >> numbits[1]);
    bigint[1] = bigint[1] & ~mask[1];
    bigint[1] += bigint[0] >> numbits[0];
    bigint[0] =  bigint[0] & ~mask[0];
    __syncthreads ();

    if (threadIdx.x) bigint[0] += carry[threadIdx.x];
    shifted_carry = bigint[1] - (mask[1] >> 1);
    bigint[1] -= shifted_carry & mask[1];
    carry[threadIdx.x] = shifted_carry >> numbits[1];
    shifted_carry = bigint[0] - (mask[0] >> 1);
    bigint[0] -= shifted_carry & mask[0];
    bigint[1] += shifted_carry >> numbits[0];
    __syncthreads ();

    if (threadIdx.x == (blockDim.x - 1))
    {
      if (blockIdx.x == gridDim.x - 1) carry_out[0] = carry[threadIdx.x + 1] + carry[threadIdx.x];
      else   carry_out[blockIdx.x + 1] =  carry[threadIdx.x + 1] + carry[threadIdx.x];
    }

    if (threadIdx.x)
    {
      bigint[0] += carry[threadIdx.x - 1];
      {
          in[index + 1] = (double) bigint[1] * ttp_temp;
          ttp_temp *= -g_ttpinc[numbits[0] == g_qn[0]];
          in[index] = (double) bigint[0] * ttp_temp;
      }
      if(err_flag)
      {
        xint[index + 1] = bigint[1];
        xint[index] = bigint[0];
      }
    }
    else
    {
      data[index1] = bigint[0];
      data[index1 + 1] = bigint[1];
    }
  }
}

// Splicing kernel, 64 bit version.
__global__ void splice2 (double *out,
                        int *xint,
                        int n,
                        int threads,
                        long long int *data,
                        long long int *carry_in,
                        double *ttp1,
                        int err_flag)
{
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  const int threadID1 = threadID << 1;
  const int j = (threads * threadID) << 1;
  long long int shifted_carry, temp0, temp1;
  int mask,  numbits = g_qn[0];
  double temp;

  if (j < n)
    {
      temp0 = data[threadID1] + carry_in[threadID];
      temp1 = data[threadID1 + 1];
      temp = ttp1[threadID];
      if(temp < 0.0)
      {
        numbits++;
        temp = -temp;
      }
      mask = -1 << numbits;
      shifted_carry = temp0 - (mask >> 1) ;
      temp0 = temp0 - (shifted_carry & mask);
      temp1 = temp1 + (shifted_carry >> numbits);
      out[j + 1] = temp1 * temp;
      temp *= -g_ttpinc[numbits == g_qn[0]];
      out[j] = temp0 * temp;
      if(err_flag)
      {
        xint[j + 1] = temp1;
        xint[j] = temp0;
      }
    }
}

/**************************************************************
***************************************************************
*                                                             *
*                        Host functions                       *
*                                                             *
***************************************************************
**************************************************************/

/**************************************************************
*                                                             *
*                        Initialization                       *
*                                                             *
**************************************************************/

void init_device (int device_number, int show_prop)
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
  cudaGetDeviceProperties (&g_dev, device_number);
  // From Iain
  if (g_dev.major == 1 && g_dev.minor < 3)
  {
    printf("A GPU with compute capability >= 1.3 is required for double precision arithmetic\n\n");
	  printf("See http://www.mersenne.ca/cudalucas.php for a list of cards\n\n");
    exit (2);
  }
  if (show_prop)
  {
    printf ("------- DEVICE %d -------\n",    device_number);
    printf ("name                %s\n",       g_dev.name);
    printf ("Compatibility       %d.%d\n",    g_dev.major, g_dev.minor);
    printf ("clockRate (MHz)     %d\n",       g_dev.clockRate/1000);
    printf ("memClockRate (MHz)  %d\n",       g_dev.memoryClockRate/1000);
    printf ("totalGlobalMem      %llu\n",     (unsigned long long) g_dev.totalGlobalMem);
    printf ("totalConstMem       %llu\n",     (unsigned long long) g_dev.totalConstMem);
    printf ("l2CacheSize         %d\n",       g_dev.l2CacheSize);
    printf ("sharedMemPerBlock   %llu\n",     (unsigned long long) g_dev.sharedMemPerBlock);
    printf ("regsPerBlock        %d\n",       g_dev.regsPerBlock);
    printf ("warpSize            %d\n",       g_dev.warpSize);
    printf ("memPitch            %llu\n",     (unsigned long long) g_dev.memPitch);
    printf ("maxThreadsPerBlock  %d\n",       g_dev.maxThreadsPerBlock);
    printf ("maxThreadsPerMP     %d\n",       g_dev.maxThreadsPerMultiProcessor);
    printf ("multiProcessorCount %d\n",       g_dev.multiProcessorCount);
    printf ("maxThreadsDim[3]    %d,%d,%d\n", g_dev.maxThreadsDim[0], g_dev.maxThreadsDim[1], g_dev.maxThreadsDim[2]);
    printf ("maxGridSize[3]      %d,%d,%d\n", g_dev.maxGridSize[0], g_dev.maxGridSize[1], g_dev.maxGridSize[2]);
    printf ("textureAlignment    %llu\n",     (unsigned long long) g_dev.textureAlignment);
    printf ("deviceOverlap       %d\n\n",     g_dev.deviceOverlap);
  }
}

void init_threads(int n)
{
  FILE *threadf;
  char buf[192];
  char threadfile[32];
  int th0 = 0, th1 = 0, th2 = 0;
  int temp;
  int i, j, k;

  sprintf (threadfile, "%s threads.txt", g_dev.name);
  threadf = fopen(threadfile, "r");
  if(threadf && g_th < 0)
  {
    while(fgets(buf, 192, threadf) != NULL)
    {
      sscanf(buf, "%d %d %d %d", &temp, &th0, &th1, &th2);
      if(n == temp * 1024)
      {
        g_thr[0] = th0;
        g_thr[1] = th1;
        g_thr[2] = th2;
      }
    }
    fclose(threadf);
  }
  temp = g_dev.maxThreadsPerBlock;
  k = -1;
  while(temp)
  {
    temp >>= 1;
    k++;
  }
  temp = n; //full fft length
  i = 0; j = 0;
  while((temp & 1) == 0)
  {
    if(temp > g_dev.maxGridSize[0]) j++;
    temp >>= 1;
    i++;
  }
  if(j == i)
  {
    while(temp > g_dev.maxGridSize[0])
    {
      temp >>= 1;
      j++;
    }
  }
  if(j < 6) j = 6;
  th0 = 1 << (i-1);
  th1 = 1 << (j-1);
  temp = 1 << k;
  if(k < j - 1)
  {
    fprintf(stderr, "The fft %dK is too big for this device.\n", n / 1024);
    exit(2);
  }
  if (i < j)
  {
    fprintf(stderr, "An fft of that size must be divisible by %d\n",(1 << j));
    exit(2);
  }
  for(i = 0; i < 3; i++)
  {
    if (g_thr[i] > temp)
    {
      fprintf(stderr, "threads[%d] = %d must be no more than %d, changing to %d.\n", i, g_thr[i], temp, temp);
      g_thr[i] = temp;
    }
    if (g_thr[i] > th0)
    {
      fprintf (stderr, "fft length %d must be divisible by %d * threads[%d] = %d\n", n,  2 * i + 2, i, (2 * i + 2) * g_thr[i]);
      fprintf(stderr, "Decreasing threads[%d] to %d\n", i, th0);
      g_thr[i] = th0;
    }
    if(g_thr[i] < 32)
    {
      fprintf(stderr, "threads[%d] = %d must be at least 32, changing to %d.\n", i,  g_thr[i], 32);
      g_thr[i] = 32;
    }
    if(g_thr[i] < th1)
    {
      fprintf (stderr, "fft length / (%d * threads[%d]) = %d must be less than the max block size = %d\n",
                        2 * i + 2, i, n / ((2 * i + 2) * g_thr[0]), g_dev.maxGridSize[0]);
      fprintf(stderr, "Increasing threads[%d] to %d\n", i, th1);
      g_thr[i] = th1;
    }
    j = 1;
    while(j < g_thr[i]) j <<= 1;
    if(j != g_thr[i])
    {
      k = j - g_thr[i];
      if(k > (j >> 2)) j >>= 1;
      fprintf(stderr, "threads[%d] = %d must be a power of two, changing to %d.\n", i, g_thr[i], j);
      g_thr[i] = j;
    }
    if(i == 0)
    {
      th0 >>= 1;
      if(j > 6) th1 >>= 1;
    }
    if(i == 1)
    {
      th0 = temp;
      th1 = 32;
    }
  }
  printf("Using threads: norm1 %d, mult %d, norm2 %d.\n", g_thr[0], g_thr[1], g_thr[2]);
  fflush(stderr);
  return;
}

int init_ffts(int new_len)
{
  #define COUNT 162
  FILE *fft;
  char buf[192];
  char fftfile[32];
  int next_fft, j = 0, i = 0, k = 1;
  int temp_fft[256] = {0};
  int default_mult[COUNT] = {  //this batch from GTX570 timings
                                1, 2,  4,  8,  10,  14,    16,    18,    20,    32,    36,    42,
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
                            52488, 54432, 55296, 56000, 57344, 60750, 62500, 64000, 64800, 65536};


  sprintf (fftfile, "%s fft.txt", g_dev.name);
  fft = fopen(fftfile, "r");
  if(fft)
  {
    while(fgets(buf, 192, fft) != NULL)
    {
      if(next_fft = atoi(buf))
      {
        next_fft <<= 10;
        if((new_len > temp_fft[i]) && (new_len < next_fft))
        {
          i++;
          temp_fft[i] = new_len;
        }
        i++;
        temp_fft[i] = next_fft;
      }
    }
    fclose(fft);
  }
  else
  {
    i = 2;
    temp_fft[1] = new_len;
  }
  while((j < COUNT) && (1024 * default_mult[j] < temp_fft[1]))
  {
    g_ffts[j] = default_mult[j] << 10;
    j++;
  }
  while(k < i && j < 256)
  {
    g_ffts[j] = temp_fft[k];
    k++;
    j++;
  }
  k = 0;
  if (j) while(k < COUNT && default_mult[k] * 1024 <= g_ffts[j - 1]) k++;
  while(k < COUNT && j < 256)
  {
    g_ffts[j] = default_mult[k] << 10;
    j++;
    k++;
  }

  //for(i = 0; i < j; i++) printf("%d %d\n", i, g_ffts[i]/1024);
  return j;
}

int choose_fft_length (int q, int *index)
{
/* In order to increase length if an exponent has a round off issue, we use an
extra paramter that we can adjust on the fly. In check(), index starts as -1,
the default. In that case, choose from the table. If index >= 0, we must assume
it's an override index and return the corresponding length. If index > table-count,
then we assume it's a manual fftlen and return the proper index. */
  double ff1 = 1.0 * 1024.0 * 0.0000357822505975293;
  double ff2 = 1.0 * 1024.0 * 0.0002670641830112380;
  double e1 = 1.022179977969700;
  double e2 = 0.929905288591965;
  int lb;
  int ub;
  int i = 0;

  if(q > 0)
  {
    lb = (int) ceil (ff1 * exp (e1 * log ((double) q)));
    ub = (int) floor(ff2 * exp (e2 * log ((double) q)));
    g_lbi = 0;
    while( g_lbi < g_fft_count && g_ffts[g_lbi] < lb) g_lbi++;
    g_ubi = g_lbi;
    while( g_ubi < g_fft_count && g_ffts[g_ubi] <= ub) g_ubi++;
    g_ubi--;
  }
  //printf("Index: %d, Lower bound at %d: %dK, Upper bound at %d: %dK\n", *index, g_lbi, g_ffts[g_lbi]/1024, g_ubi, g_ffts[g_ubi]/1024);

  if(*index >= g_fft_count) while(i < g_fft_count && g_ffts[i] < *index) i++;
  else i = *index;
  if(i < g_lbi)
  {
    if(*index) printf("The fft length %dK is too small for exponent %d, increasing to %dK\n", g_ffts[i] / 1024, q, g_ffts[g_lbi] / 1024);
    i = g_lbi;
  }
  if(i > g_ubi)
  {
    printf("The fft length %dK is too large for exponent %d, decreasing to %dK\n", g_ffts[i] / 1024, q, g_ffts[g_ubi] / 1024);
    i = g_ubi;
  }
  *index = i;
  return g_ffts[i];
}

int fft_from_str(const char* str)
/* This is really just strtoul with some extra magic to deal with K or M */
{
  char* endptr;
  const char* ptr = str;
  int len, mult = 1;
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

  len = (int) strtoul(str, &endptr, 10)*mult;
  if( endptr != ptr ) { // The K or M must directly follow the num (or the num must extend to the end of the str)
    fprintf (stderr, "can't parse fft length \"%s\"\n\n", str);
    exit (2);
  }
  return len;
}

void alloc_gpu_mem(int n)
{
  int size_d = n / 32 * 73;
  int size_i = n / 32 * 35;
  int a, c = g_dev.maxThreadsPerBlock;

  cutilSafeCall (cudaMalloc ((void **) &g_x, sizeof (double) * size_d));
  cutilSafeCall (cudaMalloc ((void **) &g_xint, sizeof (int) * size_i));
  g_data = &g_xint[n];
  g_carry = &g_xint[n / 16 * 17];
  g_ttmp = &g_x[n];
  g_ct = &g_x[2 * n];
  g_ttp1 = &g_x[n / 4 * 9];
  a = n >> 1;
  while(a & (c - 1)) c >>= 1;
  a = n * ((c >> 4) - 1) / (c << 1);
  g_err = (float *) &g_x[n / 4 * 9 + a];
}

void write_gpu_data(int q, int n)
{
  double *s_ttmp, *s_ttp1, *s_ct;
  double d;
  double *h_ttpinc;
  int *h_qn;
  int j;
  int qn = q / n, b = q % n, c = n - b, a = 0, bj = 0;

  g_size = (char *) malloc (sizeof (char) * n);
  s_ct = (double *) malloc (sizeof (double) * (n / 4));
  s_ttmp = (double *) malloc (sizeof (double) * (n));
  s_ttp1 = (double *) malloc (sizeof (double) * 64 * n / 2048);
	h_ttpinc = (double*) malloc(2 * sizeof(double));
	h_qn = (int*) malloc(1 * sizeof(int));

  for (j = 0; j < n; j++)
  {
    if(j % 2 == 0) s_ttmp[j] = exp2 (a / (double) n) * 2.0 / n;
    else s_ttmp[j] = exp2 (-a / (double) n);
    g_size[j] = (bj >= c);
    if(g_size[j]) s_ttmp[j] = -s_ttmp[j];
    bj += b;
    bj %= n;
    a = bj - n;
  }
  s_ttmp[0] = -s_ttmp[0];
  if(g_size[n-1]) s_ttmp[n-1] = -s_ttmp[n-1];
  g_size[0] = 1;
  g_size[n-1] = 0;

  h_ttpinc[0] = -exp2((b - n) / (double) n);
  h_ttpinc[1] = -exp2(b / (double) n);
  set_ttpinc(h_ttpinc);
  h_qn[0] = qn;
  set_qn(h_qn);

  a = n >> 1;
  bj = g_dev.maxThreadsPerBlock;
  while(a & (bj - 1)) bj >>= 1;
  a = 0;
  b = 32;
  while(b <= bj)
  {
    b *= 2;
    for (c = 0, j = 0; c < n ; c += b)
    {
      s_ttp1[a + j] = s_ttmp[c + 1];
      if(g_size[c] != g_size[c+1]) s_ttp1[a + j] = -s_ttp1[a + j];
      j++;
    }
    a += j;
  }

  d = 1.0 / (double) (n >> 1);
  b = n >> 2;
  for (j = 1; j < b; j++) s_ct[j] = 0.5 * cospi (j * d);

  cudaMemset (g_err, 0, sizeof (float));
  cudaMemcpy (g_ttmp, s_ttmp, sizeof (double) * n, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ttp1, s_ttp1, sizeof (double) * a, cudaMemcpyHostToDevice);
  cudaMemcpy (g_ct, s_ct, sizeof (double) * (n / 4), cudaMemcpyHostToDevice);
  cufftSafeCall (cufftPlan1d (&g_plan, n / 2, CUFFT_Z2Z, 1));

  b = n * ((g_thr[0] >> 5) - 1) / (g_thr[0]);
  g_ttp1 = &g_x[n / 4 * 9 + b];

  free ((char *) s_ct);
  free ((char *) s_ttmp);
  free ((char *) s_ttp1);
  free ((char *) h_ttpinc);
  free ((char *) h_qn);
}

void init_x(int *x_int, unsigned *x_packed, int q, int n, int *offset)
{
  int j;
  int digit, bit;
  int end = (q + 31) / 32;
  if(*offset < 0)
  {
    srand(time(0));
    *offset = rand() % q;
    bit = (*offset + 2) % q;
    digit = floor(bit * (n / (double) q));
    bit = bit - ceil(digit * (q / (double) n));
    for(j = 0; j < n; j++) x_int[j] = 0;
    x_int[digit] = (1 << bit);
    if(x_packed)
    {
      for(j = 0; j < end; j++) x_packed[j] = 0;
      x_packed[*offset / 32] = (1 << (*offset % 32));
      x_packed[end] = q;
      x_packed[end + 1] = n;
    }
  }
  cudaMemcpy (g_xint, x_int, sizeof (int) * n , cudaMemcpyHostToDevice);
  apply_weights <<<n / (2 * g_thr[0]), g_thr[0]>>> (g_x, g_xint, g_ttmp);
}

void unpack_bits(int *x_int, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;
  int mask1 = -1 << (qn + 1);
  int mask2;
  int mask;

  mask1 = ~mask1;
  mask2 = mask1 >> 1;
  for(i = 0; i < n; i++)
  {
    if(k < qn + g_size[i])
    {
      temp1 = packed_x[j];
      temp2 += (temp1 << k);
      k += 32;
      j++;
    }
    if(g_size[i]) mask = mask1;
    else mask = mask2;
    x_int[i] = ((int) temp2) & mask;
    temp2 >>= (qn + g_size[i]);
    k -= (qn + g_size[i]);
  }
}

void balance_digits(int* x, int q, int n)
{
  int half_low = (1 << (q / n - 1));
  int low = half_low << 1;
  int high = low << 1;
  int upper, adj, carry = 0;
  int j;

  for(j = 0; j < n; j++)
  {
    if(g_size[j])
    {
      upper = low;
      adj = high;
    }
    else
    {
      upper = half_low;
      adj = low;
    }
    x[j] += carry;
    carry = 0;
    if(x[j] >= upper)
    {
      x[j] -= adj;
      carry = 1;
    }
  }
  x[0] += carry; // Good enough for our purposes.
}

int *init_lucas(unsigned *x_packed,
                           int       q,
                           int       *n,
                           int       *j,
                           int       *offset,
                           unsigned long long *total_time,
                           unsigned long long  *time_adj,
                           unsigned  *iter_adj)
{
  int *x_int;
  int end = (q + 31) / 32;
  int new_test = 0;

  *n = x_packed[end + 1];
  if(*n == 0) new_test = 1;
  *j = x_packed[end + 2];
  *offset = x_packed[end + 3];
  if(total_time) *total_time = (unsigned long long) x_packed[end + 4] << 15;
  if(time_adj) *time_adj = (unsigned long long) x_packed[end + 5] << 15;
  if(iter_adj) *iter_adj = x_packed[end + 6];
  if(g_fftlen == 0) g_fftlen = *n;
  if(g_fft_count == 0) g_fft_count = init_ffts(g_fftlen);
  *n = choose_fft_length(q, &g_fftlen);
  if(*n != (int) x_packed[end + 1])
  {
    *time_adj = *total_time;
    if(*j > 1) *iter_adj = *j;
  }
  //printf("time_adj: %llu, iter_adj: %u\n", *time_adj, *iter_adj);
  init_threads(*n);

  x_int = (int *) malloc (sizeof (int) * *n);
  alloc_gpu_mem(*n);
  write_gpu_data(q, *n);
  if(!new_test)
  {
    unpack_bits(x_int, x_packed, q, *n);
    balance_digits(x_int, q, *n);
  }
  init_x(x_int, x_packed, q, *n, offset);
  return x_int;
}

/**************************************************************************
 *                                                                        *
 *                               Cleanup                                  *
 *                                                                        *
 **************************************************************************/

void free_host (int *x)
{
  free ((char *) g_size);
  free ((char *) x);
}

void free_gpu(void)
{
  cufftSafeCall (cufftDestroy (g_plan));
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_xint));
}

void close_lucas (int *x)
{
  free_host(x);
  free_gpu();
}

/**************************************************************************
 *                                                                        *
 *                        Checkpoint Processing                           *
 *                                                                        *
 **************************************************************************/


//From apsen
void print_time_from_seconds (unsigned in_sec, char *res, int mode)
{
  char s_time[32] = {0};
  unsigned day = 0;
  unsigned hour;
  unsigned min;
  unsigned sec;

  if(mode >= 2)
  {
    day = in_sec / 86400;
    in_sec %= 86400;
  }
  hour = in_sec / 3600;
  in_sec %= 3600;
  min = in_sec / 60;
  sec = in_sec % 60;
  if (day)
  {
    if(mode & 1) sprintf (s_time, "%ud%02uh%02um%02us", day, hour, min, sec);
    else sprintf (s_time, "%u:%02u:%02u:%02u", day, hour, min, sec);
  }
  else if (hour)
  {
     if(mode & 1) sprintf (s_time, "%uh%02um%02us", hour, min, sec);
     else sprintf (s_time, "%u:%02u:%02u", hour, min, sec);
  }
  else
  {
     if(mode & 1) sprintf (s_time, "%um%02us", min, sec);
     else sprintf (s_time, "%u:%02u", min, sec);
  }
  if(res) sprintf(res, "%12s", s_time);
  else printf ("%s", s_time);
}

int standardize_digits(int *x_int, int q, int n, int offset, int num_digits)
{
  int j, digit, stop, qn = q / n, carry = 0;
  int temp;
  int lo = 1 << qn;
  int hi = lo << 1;

  digit = floor(offset * (n / (double) q));
  j = (n + digit - 1) % n;
  while(x_int[j] == 0 && j != digit) j = (n + j - 1) % n;
  if(j == digit && x_int[digit] == 0) return(1);
  else if (x_int[j] < 0) carry = -1;
  {
    stop = (digit + num_digits) % n;
    j = digit;
    do
    {
      x_int[j] += carry;
      carry = 0;
      if (g_size[j]) temp = hi;
      else temp = lo;
      if(x_int[j] < 0)
      {
        x_int[j] += temp;
        carry = -1;
      }
      j = (j + 1) % n;
    }
    while(j != stop);
  }
  return(0);
}

unsigned long long find_residue(int *x_int, int q, int n, int offset)
{
  int j, k = 0;
  int digit, bit;
  unsigned long long temp, residue = 0;

  digit = floor(offset *  (n / (double) q));
  bit = offset - ceil(digit * (q / (double) n));
  j = digit;
  while(k < 64)
  {
    temp = x_int[j];
    residue = residue + (temp << k);
    k += q / n + g_size[j % n];
    if(j == digit)
    {
       k -= bit;
       residue >>= bit;
    }
    j = (j + 1) % n;
  }
  return residue;
}

void printbits(unsigned long long residue, int q, int n, int offset, FILE* fp, int o_f)
{
  if (fp)
  {
    printf ("M( %d )C, 0x%016llx,", q, residue);
    fprintf (fp, "M( %d )C, 0x%016llx,", q, residue);
    if(o_f) fprintf(fp, " offset = %d,", offset);
    fprintf (fp, " n = %dK, %s", n/1024, program);
  }
  else printf ("/ %d, 0x%016llx,", q, residue);
  printf (" %dK, %s", n/1024, program);
}

void pack_bits(int *x_int, unsigned *packed_x, int q , int n)
{
  unsigned long long temp1, temp2 = 0;
  int i, j = 0, k = 0;
  int qn = q / n;

  for(i = 0; i < n; i++)
  {
    temp1 = x_int[i];
    temp2 += (temp1 << k);
    k += qn + g_size[i];
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

void set_checkpoint_data(unsigned *x_packed, int q, int j, int offset, unsigned long long time, unsigned long long t_adj, unsigned i_adj)
{
  int end = (q + 31) / 32;

  x_packed[end + 2] = j;
  x_packed[end + 3] = offset;
  x_packed[end + 4] = time >> 15;
  x_packed[end + 5] = t_adj >> 15;
  x_packed[end + 6] = i_adj;
}

void reset_err(float* maxerr, float value)
{
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  *maxerr *= value;
}

void process_output(int q,
                    int n,
                    int j,
                    int offset,
                    int last_chk,
                    float maxerr,
                    unsigned long long residue,
                    unsigned long long diff,
                    unsigned long long diff1,
                    unsigned long long diff2,
                    unsigned long long total_time)
{
  time_t now;
  struct tm *tm_now = NULL;
  int index = 0;
  char buffer[192];
  char temp [32];
  int i;
  static int header_iter = 0;

  now = time(NULL);
  tm_now = localtime(&now);
  strftime(g_output[0], 32, "%a", tm_now);                             //day of the week Sat
  strftime(g_output[1], 32, "%b", tm_now);                             //month
  strftime(g_output[2], 32, "%d", tm_now);                             //date
  strftime(g_output[3], 32, "%H", tm_now);                             //hour
  strftime(g_output[4], 32, ":%M", tm_now);                            //min
  strftime(g_output[5], 32, ":%S", tm_now);                            //sec
  strftime(g_output[6], 32, "%y", tm_now);                             //year
  strftime(g_output[7], 32, "%Z", tm_now);                             //time zone
  sprintf(g_output[8],  "%9d", q);                                      //exponent
  sprintf(temp,  "M%d", q);
  sprintf(g_output[9],  "%10s", temp);                                 //Mersenne number
  sprintf(g_output[10], "%5dK", n / 1024);                             //fft, multiple of 1k
  sprintf(g_output[11], "%8d", n);                                     //fft
  sprintf(g_output[12], "%9d", j);                                     //iteration
  sprintf(g_output[13], "%9d", offset);                                //offset
  sprintf(g_output[14], "0x%016llx", residue);                         //residue, lower case hex
  sprintf(g_output[15], "0x%016llX", residue);                         //residue, upper case hex
  sprintf(g_output[18], "%s", program);                                //program name
  sprintf(g_output[19], "%%");                                         //percent done
  sprintf(g_output[25], "%7.5f", maxerr);                              //round off error
  sprintf(g_output[26], "%9.5f", diff1 / 1000.0 / (j - last_chk));     //ms/iter
  sprintf(g_output[27], "%10.5f", diff1 / 1000000.0 );                 //time since last checkpoint xxx.xxxxx seconds format
  sprintf(g_output[28], "%10.5f", (j - last_chk) * 1000000.0 / diff1); //iter/sec
  sprintf(g_output[29], "%10.5f", j / (float) (q - 2) * 100 );         //percent done
  print_time_from_seconds(diff, g_output[16], 0);                      //time since last checkpoint hh:mm:ss format
  print_time_from_seconds(diff2, g_output[17], 2);                     //ETA in d:hh:mm:ss format

  i = 0;
  while( i < 50 && g_output_code[i] >= 0)
  {
    index += sprintf(buffer + index,"%s",g_output[g_output_code[i]]);
    if(g_output_code[i] >= 25 && g_output_code[i] < 30)
    {
      i++;
      index -= g_output_code[i];
    }
    i++;
  }
  if(header_iter == 0 && g_output_interval > 0)
  {
    header_iter = g_output_interval - 1;
    printf("%s\n", g_output_header);
  }
  header_iter--;
  printf("%s\n", buffer);
 }

/**************************************************************************
 *                                                                        *
 *                        File Related Functions                          *
 *                                                                        *
 **************************************************************************/
unsigned magic_number(unsigned *x_packed, int q)
{
  return 0;
}

unsigned ch_sum(unsigned *x_packed, int q)
{
  int end = (q + 31) / 32;
  int j;
  unsigned sum = 0;

  for(j = 0; j < end + 9; j++) sum += x_packed[j];
  return sum;
}


unsigned *read_checkpoint(int q)
{
  FILE *fPtr;
  unsigned *x_packed;
  char chkpnt_cfn[32];
  char chkpnt_tfn[32];
  int end = (q + 31) / 32;
  int i;

  x_packed = (unsigned *) malloc (sizeof (unsigned) * (end + 10));
  for(i = 0; i < 10; i++) x_packed[end + i] = 0;
  x_packed[end + 2] = 1;
  x_packed[end + 3] = (unsigned) -1;
  if(g_rt) return(x_packed);

  sprintf (chkpnt_cfn, "c%d", q);
  sprintf (chkpnt_tfn, "t%d", q);
  fPtr = fopen (chkpnt_cfn, "rb");
  if (fPtr)
  {
    if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
    {
      fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
      fclose (fPtr);
    }
    else
    {
      fclose(fPtr);
      if(x_packed[end + 9] != ch_sum(x_packed, q)) fprintf (stderr, "\nThe checkpoint appears to be corrupt. Trying the backup file.\n");
      //else
      return x_packed;
    }
  }
  fPtr = fopen(chkpnt_tfn, "rb");
  if (fPtr)
  {
    if (fread (x_packed, 1, sizeof (unsigned) * (end + 10) , fPtr) != (sizeof (unsigned) * (end + 10)))
    {
      fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
      fclose (fPtr);
    }
    else
    {
      fclose(fPtr);
      if(x_packed[end + 9] != ch_sum(x_packed, q)) fprintf (stderr, "\nThe backup appears to be corrupt. Restarting test.\n");
      //else
      return x_packed;
    }
  }
  return x_packed;
}

void write_checkpoint(unsigned *x_packed, int q, unsigned long long residue)
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
  x_packed[end + 8] = magic_number(x_packed, q);
  x_packed[end + 9] = ch_sum(x_packed, q);;
  fwrite (x_packed, 1, sizeof (unsigned) * (end + 10), fPtr);
  fclose (fPtr);
  if (g_sf > 0)			// save all checkpoint files
    {
      char chkpnt_sfn[64];
#ifndef _MSC_VER
      sprintf (chkpnt_sfn, "%s/s" "%d.%d.%016llx", g_folder, q, x_packed[end + 2], residue);
#else
      sprintf (chkpnt_sfn, "%s\\s" "%d.%d.%016llx.txt", g_folder, q, x_packed[end + 2], residue);
#endif
      fPtr = fopen (chkpnt_sfn, "wb");
      if (!fPtr) return;
      fwrite (x_packed, 1, sizeof (unsigned) * (((q + 31) / 32) + 10), fPtr);
      fclose (fPtr);
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

/**************************************************************************
 *                                                                        *
 *                        LL Test Iteration Loop                          *
 *                                                                        *
 **************************************************************************/

float lucas_square (int q, int n, int iter, int last, int* offset, float* maxerr, int error_flag)
{
  int digit, bit;
  float terr = 0.0;
  cudaError err = cudaSuccess;

  *offset = (2 * *offset) % q;
  bit = (*offset + 1) % q;
  digit = floor(bit * (n / (double) q));
  bit = bit - ceil(digit * (q / (double) n));
  bit = 1 << bit;

  cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
  cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  if(error_flag & 4)
  {
    rcb2 <<<n / (2 * g_thr[0]), g_thr[0] >>>
             (g_x, g_xint, (long long *) g_data, g_ttmp, (long long *) g_carry, g_err, *maxerr, digit, bit, error_flag & 1);
    splice2 <<< (n / (2 * g_thr[0]) + g_thr[2] - 1) / g_thr[2], g_thr[2] >>>
             (g_x, g_xint, n, g_thr[0], (long long *) g_data, (long long *) g_carry, g_ttp1, error_flag & 1);
  }
  else
  {
    rcb1 <<<n / (2 * g_thr[0]), g_thr[0] >>> (g_x, g_xint, g_data, g_ttmp, g_carry, g_err, *maxerr, digit, bit, error_flag & 1);
    splice1 <<< (n / (2 * g_thr[0]) + g_thr[2] - 1) / g_thr[2], g_thr[2] >>>
             (g_x, g_xint, n, g_thr[0], g_data, g_carry, g_ttp1, error_flag & 1);
  }
  if (error_flag & 3)
  {
    err = cutilSafeCall1 (cudaMemcpy (&terr, g_err, sizeof (float), cudaMemcpyDeviceToHost));
    if(terr > *maxerr) *maxerr = terr;
  }
  else if (g_pf && (iter % g_po) == 0) err = cutilSafeThreadSync();
  if(err != cudaSuccess) terr = -1.0;
  return (terr);
}

/**************************************************************************
 *                                                                        *
 *                        Benchmarking and Testing                        *
 *                                                                        *
 **************************************************************************/
int isReasonable(int fft)
{ //From an idea of AXN's mentioned on the forums
  int i;

  while(!(fft & 1)) fft >>= 1;
  for(i = 3; i <= 7; i += 2) while((fft % i) == 0) fft /= i;
  return (fft);
}

void kernel_percentages (int fft, int passes, int device_number)
{
  int pass, i, j;
  int n = fft * 1024;
  float total0[4] = {0.0f};
  float total1[4] = {0.0f};
  float maxerr = 0.0f, outerTime, t0, t1;
  cudaEvent_t start, stop;

  alloc_gpu_mem(n);
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * n / 4));

  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
  cufftSafeCall (cufftPlan1d (&g_plan, n / 2, CUFFT_Z2Z, 1));

  init_threads(n);
  for(j = 0; j < 4; j++)
  {
    for(pass = 0; pass < passes; pass++)
    {
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 50; i++)
      {
        if(j == 0)
        {
         rcb1 <<<n / (2 * g_thr[0]), g_thr[0] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
        }
        if(j == 1)
        {
         rcb1 <<<n / (2 * g_thr[0]), g_thr[0] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
         splice1 <<< (n / (2 * g_thr[0]) + g_thr[2] - 1) / g_thr[2], g_thr[2] >>> (g_x, NULL, n, g_thr[0], g_data, g_carry, g_ttp1, 0);
        }
        if(j == 2)
        {
         square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
         rcb1 <<<n / (2 * g_thr[0]), g_thr[0] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
         splice1 <<< (n / (2 * g_thr[0]) + g_thr[2] - 1) / g_thr[2], g_thr[2] >>> (g_x, NULL, n, g_thr[0], g_data, g_carry, g_ttp1, 0);
        }
        if(j == 3)
        {
         cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
         square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
         cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
         rcb1 <<<n / (2 * g_thr[0]), g_thr[0] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
         splice1 <<< (n / (2 * g_thr[0]) + g_thr[2] - 1) / g_thr[2], g_thr[2] >>> (g_x, NULL, n, g_thr[0], g_data, g_carry, g_ttp1, 0);
        }
      }
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      outerTime /= 50.0f;
      total0[j] += outerTime;
    }
  }
  for(j = 0; j < 4; j++)
  {
    for(pass = 0; pass < passes; pass++)
    {
      cutilSafeCall (cudaEventRecord (start, 0));
      for (i = 0; i < 50; i++)
      {
        if(j == 0)
        {
         rcb1 <<<n / (2 * g_thr[0]), g_thr[0] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
        }
        if(j == 1)
        {
         splice1 <<< (n / (2 * g_thr[0]) + g_thr[2] - 1) / g_thr[2], g_thr[2] >>> (g_x, NULL, n, g_thr[0], g_data, g_carry, g_ttp1, 0);
        }
        if(j == 2)
        {
         square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
        }
        if(j == 3)
        {
         cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
         cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
       }
      }
      cutilSafeCall (cudaEventRecord (stop, 0));
      cutilSafeCall (cudaEventSynchronize (stop));
      cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
      outerTime /= 50.0f;
      total1[j] += outerTime;
    }
  }
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));
  free_gpu();

  t0 = total0[3];
  t1 = total1[0] + total1[1] + total1[2] + total1[3];
  printf("Total time: %11.5f, %8.5f, %6.2f\n", t0/passes, t1/passes, 100.0);
  printf("RCB Kernel: %11.5f, %8.5f, %6.2f\n", total0[0]/passes, total1[0]/passes, total0[0] / t0 * 100.0);
  printf("Splice Kernel: %8.5f, %8.5f, %6.2f\n", (total0[1] - total0[0])/passes, total1[1]/passes, (total0[1] - total0[0]) / t0 * 100.0);
  printf("Mult Kernel: %10.5f, %8.5f, %6.2f\n", (total0[2] - total0[1])/passes, total1[2]/passes, (total0[2] - total0[1]) / t0 * 100.0);
  printf("FFTs: %17.5f, %8.5f, %6.2f\n", (total0[3] - total0[2])/passes, total1[3]/passes, (total0[3] - total0[2]) / t0 * 100.0);
}



void threadbench (int st_fft, int end_fft, int passes, int device_number)
{
  float total[36] = {0.0f}, outerTime, maxerr = 0.5f;
  int th[10];
  int t1, t2, t3, i, j;
  float best_time = 0.0f;
  int best_t1 = 0, best_t2 = 0, best_t3 = 0;
  int results[256][4];
  int pass;
  int n = end_fft << 10;
  int fft;
  cudaEvent_t start, stop;

  if(st_fft == end_fft) printf("Thread bench, testing various thread sizes for fft %dK, doing %d passes.\n", end_fft, passes);
  else printf("Thread bench, testing various thread sizes for ffts %dK to %dK, doing %d passes.\n", st_fft, end_fft, passes);
  fflush(NULL);

  alloc_gpu_mem(n);
  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));

  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * n / 4));
  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));

  th[0] = 32;
  for(i = 0; i < 10; i++)
  {
    j = th[i] << 1;
    if(j <= g_dev.maxThreadsPerBlock) th[i+1] = j;
  }
  for(i = 0; i < 256; i++) results[i][0] = 0;
  g_fft_count = init_ffts(0);
  j = st_fft * 1024;
  fft = choose_fft_length(0, &j);
  t2 = 2;
  while(fft <= n)
  {
    cufftSafeCall (cufftPlan1d (&g_plan, fft / 2, CUFFT_Z2Z, 1));
    for(t1 = 0; t1 < 6; t1++)
    {
      if(fft / (2 * th[t1]) <= g_dev.maxGridSize[0] && fft % (2 * th[t1]) == 0)
      {
       for (t3 = 0; t3 < 6; t3++)
        {
          for(pass = 1; pass <= passes; pass++)
          {
            cutilSafeCall (cudaEventRecord (start, 0));
            for (i = 0; i < 50; i++)
            {
               cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
               square <<< fft / (4 * th[t2]), th[t2] >>> (fft, g_x, g_ct);
               cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
               rcb1 <<<fft / (2 * th[t1]), th[t1] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
               splice1 <<< (fft / (2 * th[t1]) + th[t3] - 1) / th[t3], th[t3] >>> (g_x, NULL, fft, th[t1], g_data, g_carry, g_ttp1, 0);
            }
            cutilSafeCall (cudaEventRecord (stop, 0));
            cutilSafeCall (cudaEventSynchronize (stop));
            cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
            outerTime /= 50.0f;
            total[6 * t1 + t3] += outerTime;
          }
          printf ("fft size = %dK, ave time = %2.4f msec, Norm1 threads %d, Norm2 threads %d\n",
                    fft / 1024 , total[6 * t1 + t3] / passes, th[t1], th[t3]);
          fflush(NULL);
        }
      }
    }
    best_time = 10000.0f;
    for (i = 0; i < 36; i++)
    {
      if(total[i] < best_time && total[i] > 0.0f)
      {
        best_time = total[i];
        best_t3 = i % 6;
        best_t1 = i / 6;
      }
    }
    for(i = 0; i < 6; i++) total[i] = 0.0f;

    t1 = best_t1;
    t3 = best_t3;
    for (t2 = 0; t2 < 6; t2++)
    {
      if(fft / (4 * th[t2]) <= g_dev.maxGridSize[0] && fft % (4 * th[t2]) == 0)
      {
        for(pass = 1; pass <= passes; pass++)
        {
          cutilSafeCall (cudaEventRecord (start, 0));
          for (i = 0; i < 50; i++)
          {
            cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
            square <<< fft / (4 * th[t2]), th[t2] >>> (fft, g_x, g_ct);
            cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
            rcb1 <<<fft / (2 * th[t1]), th[t1] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
            splice1 <<< (fft / (2 * th[t1]) + th[t3] - 1) / th[t3], th[t3] >>> (g_x, NULL, fft, th[t1], g_data, g_carry, g_ttp1, 0);
          }
          cutilSafeCall (cudaEventRecord (stop, 0));
          cutilSafeCall (cudaEventSynchronize (stop));
          cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
          outerTime /= 50.0f;
          total[t2] += outerTime;
        }
        printf ("fft size = %dK, ave time = %2.4f msec, Norm1 threads %d, Mult threads %d, Norm2 threads %d\n",
                fft / 1024 , total[t2] / passes, th[t1], th[t2], th[t3]);
        fflush(NULL);
      }
    }
    cufftSafeCall (cufftDestroy (g_plan));
    best_time = 10000.0f;
    for (i = 0; i < 6; i++)
    {
      if(total[i] < best_time && total[i] > 0.0f)
      {
        best_time = total[i];
        best_t2 = i;
      }
    }
    for(i = 0; i < 36; i++) total[i] = 0.0f;
    printf("Best time for fft = %dK, time: %2.4f, t0 = %d, t1 = %d, t2 = %d\n\n",
           fft / 1024, best_time / passes, th[best_t1], th[best_t2], th[best_t3]);
    results[j][0] = fft / 1024;
    results[j][1] = th[best_t1];
    results[j][2] = th[best_t2];
    results[j][3] = th[best_t3];
    j++;
    if(j < g_fft_count) fft = g_ffts[j];
    else fft = n + 1;
  }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_xint));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));


  char threadfile[32];
  FILE *fptr;

  sprintf (threadfile, "%s threads.txt", g_dev.name);
  fptr = fopen(threadfile, "a+");
  if(fptr)
  {
    for(j = 0; j < 256; j++)
      if(results[j][0])
        fprintf(fptr, "%5d %4d %4d %4d\n", results[j][0], results[j][1], results[j][2], results[j][3]);
    fclose(fptr);
  }
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

void memtest(int s, int iter, int device)
{
  int i, j, k, m, u, v;
  int q = 60000091;
  int n = 3200 * 1024;
  int rand_int;
  int *i_data;
  double *d_data;
  double *dev_data1;
  double *dev_data2;
  int *d_compare;
  int h_compare;
  int total = 0;
  int total_iterations;
  int iterations_done = 0;
  float percent_done = 0.0f;
  timeval time0, time1;
  unsigned long long diff;
  unsigned long long diff1;
  unsigned long long diff2;
  unsigned long long ttime = 0;
  double total_bytes;
  size_t global_mem, free_mem;

  cudaMemGetInfo(&free_mem, &global_mem);
  printf("CUDA reports %lluM of %lluM GPU memory free.\n", (unsigned long long)free_mem/1024/1024, (unsigned long long)global_mem/1024/1024);
  if((size_t) s *1024 * 1024 * 25  > free_mem )
  {
    s = free_mem / 1024 / 1024 / 25;
     printf("Reducing size to %d\n", s);
  }
  printf("\nInitializing memory test using %0.0fMB of memory on device %d...\n", n / 1024.0 * s / 1024.0 * 8.0, device);

  i_data = (int *) malloc (sizeof (int) * n);
  srand(time(0));
  for (j = 0; j < n; j++)
  {
    rand_int = rand() % (1 << 18);
    rand_int -= (1 << 17);
    i_data[j] = rand_int;
  }
  cudaMemcpy (g_xint, i_data, sizeof (int) * n, cudaMemcpyHostToDevice);
  free_host(i_data);

  alloc_gpu_mem(n);
  write_gpu_data(q, n);
  apply_weights <<<n / (2 * g_thr[0]), g_thr[0]>>> (g_x, g_xint, g_ttmp);
  d_data = (double *) malloc (sizeof (double) * n * 5);
  cudaMemcpy (d_data, g_ttmp, sizeof (double) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy (&d_data[n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);
  cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  cudaMemcpy (&d_data[2 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);
  square <<< n / (4 * g_thr[1]), g_thr[1] >>> (n, g_x, g_ct);
  cudaMemcpy (&d_data[3 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);
  cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
  cudaMemcpy (&d_data[4 * n], g_x, sizeof (double) * n, cudaMemcpyDeviceToHost);
  free_gpu();

  cutilSafeCall (cudaMalloc ((void **) &d_compare, sizeof (int)));
  cutilSafeCall (cudaMemset (d_compare, 0, sizeof (int)));
  if( s > 3) u = (s + 1) / 2;
  else u = s;
  v = s / 2;
  cutilSafeCall (cudaMalloc ((void **) &dev_data1, sizeof (double) * n * u));

  total_iterations = s * 5 * iter;
  iter *= 10000;
  printf("Beginning test.\n\n");
  fflush(NULL);
  gettimeofday (&time0, NULL);
  for(j = 0; j < u; j++)
  {
    m = (j + 1) % u;
    for(i = 0; i < 5; i++)
    {
      cutilSafeCall (cudaMemcpy (&dev_data1[j * n], &d_data[i * n], sizeof (double) * n, cudaMemcpyHostToDevice));
      for(k = 1; k <= iter; k++)
      {
        copy_kernel <<<n / 512, 512 >>> (dev_data1, n, j, m);
        compare_kernel<<<n / 512, 512>>> (&dev_data1[m * n], &dev_data1[j * n], d_compare);
        if(k%100 == 0) cutilSafeThreadSync();
        if(k%10000 == 0)
        {
          cutilSafeCall (cudaMemcpy (&h_compare, d_compare, sizeof (int), cudaMemcpyDeviceToHost));
          cutilSafeCall (cudaMemset (d_compare, 0, sizeof (int)));
          total += h_compare;
          iterations_done++;
          percent_done = iterations_done * 100 / (float) total_iterations;
          gettimeofday (&time1, NULL);
          diff = time1.tv_sec - time0.tv_sec;
          diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
          time0.tv_sec = time1.tv_sec;
          time0.tv_usec = time1.tv_usec;
          ttime += diff1;
          diff2 = ttime  * (total_iterations - iterations_done) / iterations_done / 1000000;
          total_bytes = 244140625 / (double) diff1;
          printf("Position %d, Data Type %d, Iteration %d, Errors: %d, completed %2.2f%%, Read %0.2fGB/s, Write %0.2fGB/s, ETA ", j, i, iterations_done * 100000, total, percent_done, 3.0 * total_bytes, total_bytes);
          print_time_from_seconds ((unsigned) diff2, NULL, 0);
          printf (")\n");
          fflush(NULL);
        }
      }
    }
  }
  if(s > 3)
  {
    cudaMemGetInfo(&free_mem, &global_mem);
    if((size_t) v *1024 * 1024 * 25  > free_mem )
    {
      v = free_mem / 1024 / 1024 / 25;
      printf("Reducing size to %d\n", v);
    }
    cutilSafeCall (cudaMalloc ((void **) &dev_data2, sizeof (double) * n * v));
    cutilSafeCall (cudaFree ((char *) dev_data1));
    for(j = 0; j < v; j++)
    {
      m = (j + 1) % v;
      for(i = 0; i < 5; i++)
      {
        cutilSafeCall (cudaMemcpy (&dev_data2[j * n], &d_data[i * n], sizeof (double) * n, cudaMemcpyHostToDevice));
        for(k = 1; k <= iter; k++)
        {
          copy_kernel <<<n / 512, 512 >>> (dev_data2, n, j, m);
          compare_kernel<<<n / 512, 512>>> (&dev_data2[m * n], &dev_data2[j * n], d_compare);
          if(k%100 == 0) cutilSafeThreadSync();
          if(k%10000 == 0)
          {
            cutilSafeCall (cudaMemcpy (&h_compare, d_compare, sizeof (int), cudaMemcpyDeviceToHost));
            cutilSafeCall (cudaMemset (d_compare, 0, sizeof (int)));
            total += h_compare;
            iterations_done++;
            percent_done = iterations_done * 100 / (float) total_iterations;
            gettimeofday (&time1, NULL);
            diff = time1.tv_sec - time0.tv_sec;
            diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
            time0.tv_sec = time1.tv_sec;
            time0.tv_usec = time1.tv_usec;
            ttime += diff1;
            diff2 = ttime  * (total_iterations - iterations_done) / iterations_done / 1000000;
            total_bytes = 244140625 / (double) diff1;
            printf("Position %d, Data Type %d, Iteration %d, Errors: %d, completed %2.2f%%, Read %0.2fGB/s, Write %0.2fGB/s, ETA ",
                    j, i, iterations_done * 100000, total, percent_done, 3.0 * total_bytes, total_bytes);
            print_time_from_seconds ((unsigned) diff2, NULL, 0);
            printf (")\n");
            fflush(NULL);
          }
        }
      }
    }
  }
  printf("Test complete. Total errors: %d.\n", total);
  fflush(NULL);
  if(s > 3) cutilSafeCall (cudaFree ((char *) dev_data2));
  else cutilSafeCall (cudaFree ((char *) dev_data1));
  cutilSafeCall (cudaFree ((char *) d_compare));
  free((char*) d_data);
}

void cufftbench (int cufftbench_s, int cufftbench_e, int passes, int device_number)
{

  cudaEvent_t start, stop;
  float outerTime;
  int i, j, k;
  int end = cufftbench_e - cufftbench_s + 1;
  float best_time;
  float *total, *max_diff, maxerr = 0.5f;
  int th[] = {32, 64, 128, 256, 512, 1024};
  int t1 = 3, t2 = 2, t3 = 2;
  int n = cufftbench_e << 10;
  int warning = 0;
  cudaError err = cudaSuccess;

  if(cufftbench_s == 0)
  {
    kernel_percentages (cufftbench_e, passes, device_number);
    return;
  }

  if(end < 2)
  {
    threadbench(cufftbench_e, cufftbench_s, passes, device_number);
    return;
  }

  printf ("CUDA bench, testing reasonable fft sizes %dK to %dK, doing %d passes.\n", cufftbench_s, cufftbench_e, passes);
  j = 1;
  while(j < cufftbench_s) j <<= 1;
  k = j;
  while(j < cufftbench_e) j <<= 1;
  if(k > cufftbench_s || j > cufftbench_e) warning = 1;

  total = (float *) malloc (sizeof (float) * end);
  max_diff = (float *) malloc (sizeof (float) * end);

  for(i = 0; i < end; i++)
  {
    total[i] = max_diff[i] = 0.0f;
  }

  alloc_gpu_mem(n);

  cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
  cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * n));
  cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * n / 4));

  cutilSafeCall (cudaEventCreate (&start));
  cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
  for (j = cufftbench_s; j <= cufftbench_e; j++)
  {
    if(isReasonable(j) <= 1)
    {
      n = j * 1024;
      cufftSafeCall (cufftPlan1d (&g_plan, n / 2, CUFFT_Z2Z, 1));
      for(k = 0; k < passes; k++)
      {
        cutilSafeCall (cudaEventRecord (start, 0));
        for (i = 0; i < 50; i++)
  	    {
          cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          square <<< n / (4 * th[t2]), th[t2] >>> (n, g_x, g_ct);
          cufftSafeCall (cufftExecZ2Z (g_plan, (cufftDoubleComplex *) g_x, (cufftDoubleComplex *) g_x, CUFFT_INVERSE));
          rcb1 <<<n / (2 * th[t1]), th[t1] >>> (g_x, NULL, g_data, g_ttmp, g_carry, g_err, maxerr, 0, 0, 0);
          splice1 <<< (n / (2 * th[t1]) + th[t3] - 1) / th[t3], th[t3] >>> (g_x, NULL, n, th[t1], g_data, g_carry, g_ttp1, 0);
        }
        cutilSafeCall (cudaEventRecord (stop, 0));
        err = cutilSafeCall1(cudaEventSynchronize (stop));
      	if( cudaSuccess != err)
        {
          total[j] = 0.0;
          max_diff[j] = 0.0;
          k = passes;
          j--;
          cudaDeviceReset();
          init_device(g_dn,0);
          n = cufftbench_e * 1024;
          alloc_gpu_mem(n);
          cutilSafeCall (cudaMalloc ((void **) &g_err, sizeof (float)));
          cutilSafeCall (cudaMemset (g_err, 0, sizeof (float)));
          cutilSafeCall (cudaMemset (g_x, 0, sizeof (double) * n));
          cutilSafeCall (cudaMemset (g_ttmp, 0, sizeof (double) * n));
          cutilSafeCall (cudaMemset (g_ct, 0, sizeof (double) * n / 4));

          cutilSafeCall (cudaEventCreate (&start));
          cutilSafeCall (cudaEventCreateWithFlags (&stop, cudaEventBlockingSync));
        }
        else
        {
          cutilSafeCall (cudaEventElapsedTime (&outerTime, start, stop));
          i = j - cufftbench_s;
          outerTime /= 50.0f;
          total[i] += outerTime;
          if(outerTime > max_diff[i]) max_diff[i] = outerTime;
        }
      }
     	if(cudaSuccess == err)
      {
        cufftSafeCall (cufftDestroy (g_plan));
        printf ("fft size = %dK, ave time = %6.4f msec, max-ave = %0.5f\n", j, total[i] / passes, max_diff[i] - total[i] / passes);
      }
     fflush(NULL);

    }
  }
  cutilSafeCall (cudaFree ((char *) g_x));
  cutilSafeCall (cudaFree ((char *) g_xint));
  cutilSafeCall (cudaEventDestroy (start));
  cutilSafeCall (cudaEventDestroy (stop));

  i = end - 1;
  j = 1;
  while(j < end) j <<= 1;
  j >>= 1;
  k = j - cufftbench_s;
  best_time = total[i] + 1000000.0f;
  while(i >= 0)
  {
    if(total[i] > 0.0f && total[i] < best_time) best_time = total[i];
    else if(i != k) total[i] = 0.0f;
    if(i == k)
    {
      j >>= 1;
      k = j - cufftbench_s;
    }
    i--;
  }

  char fftfile[32];
  FILE *fptr;

  sprintf (fftfile, "%s fft.txt", g_dev.name);
  fptr = fopen(fftfile, "w");
  if(!fptr)
  {
    printf("Cannot open %s.\n",fftfile);
    printf ("Device              %s\n", g_dev.name);
    printf ("Compatibility       %d.%d\n", g_dev.major, g_dev.minor);
    printf ("clockRate (MHz)     %d\n", g_dev.clockRate/1000);
    printf ("memClockRate (MHz)  %d\n", g_dev.memoryClockRate/1000);
    printf("\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
         int tl = (int) (exp(0.9784876919 * log ((double) cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        printf("%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fflush(NULL);
  }
  else
  {
    fprintf (fptr, "Device              %s\n", g_dev.name);
    fprintf (fptr, "Compatibility       %d.%d\n", g_dev.major, g_dev.minor);
    fprintf (fptr, "clockRate (MHz)     %d\n", g_dev.clockRate/1000);
    fprintf (fptr, "memClockRate (MHz)  %d\n", g_dev.memoryClockRate/1000);
    fprintf(fptr, "\n  fft    max exp  ms/iter\n");
    for(i = 0; i < end; i++)
    {
      if(total[i] > 0.0f)
      {
        int tl = (int) (exp(0.9784876919 * log ((double) cufftbench_s + i)) * 22366.92473079 / 1.01);
        if(tl % 2 == 0) tl -= 1;
        while(!isprime(tl)) tl -= 2;
        fprintf(fptr, "%5d %10d %8.4f\n", cufftbench_s + i, tl, total[i] / passes);
      }
    }
    fclose(fptr);
    if(warning) printf("\nWARNING, the bounds were not both powers of two, results at either end may not be accurate.\n\n");
    printf("Optimal fft lengths saved in %s.\nPlease email a copy to james@mersenne.ca.\n", fftfile);
    fflush(NULL);
  }

  free ((char *) total);
  free ((char *) max_diff);
}

int round_off_test(int q, int n, int *j, int *offset)
{
  int k;
  float totalerr = 0.0;
  float terr, avgerr, maxerr = 0.0;
  float max_err = 0.0, max_err1 = 0.0;
  int l_offset = *offset;
  int last = q - 2;
  int error_flag = 0;

  printf("Running careful round off test for 1000 iterations.\n");
  printf("If average error > 0.25, or maximum error > 0.35,\n");
  printf("the test will restart with a longer FFT.\n");
  fflush(NULL);
  for (k = 0; k < 1000 && (k + *j <= last); k++)
  {
    error_flag = 2;
    if (k == 999) error_flag = 1;;
    terr = lucas_square (q, n, *j + k, q - 1, &l_offset, &maxerr, error_flag);
    if(terr > maxerr) maxerr = terr;
    if(terr > max_err) max_err = terr;
    if(terr > max_err1) max_err1 = terr;
    totalerr += terr;
    reset_err(&maxerr, 0.0);
    if(terr > 0.35)
    {
      printf ("Iteration = %d < 1000 && err = %5.5f > 0.35, increasing n from %dK\n", k, terr, n/1024);
      g_fftlen++;
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
    g_fftlen++;
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
  }
  *offset = l_offset;
  *j += 1000;
  return 0;
}

void check_residue (void)
{
  int q, n, j, last = 10000, offset;
  unsigned *x_packed = NULL;
  int  *x_int = NULL;
  float terr, maxerr = 0.25;
  int restarting;
  int error_flag = 0;
  int bad_selftest = 0;
  int i;
  unsigned long long diff;
  unsigned long long diff1;
  unsigned long long expected_residue;
  unsigned long long residue;
  timeval time0, time1;

  typedef struct res_test
  {
    int exponent;
    unsigned long long residue;
  } res_test;

  res_test test[20] = {{    86243, 0x23992ccd735a03d9}, {   132049, 0x4c52a92b54635f9e},
                       {   216091, 0x30247786758b8792}, {   756839, 0x5d2cbe7cb24a109a},
                       {   859433, 0x3c4ad525c2d0aed0}, {  1257787, 0x3f45bf9bea7213ea},
                       {  1398269, 0xa4a6d2f0e34629db}, {  2976221, 0x2a7111b7f70fea2f},
                       {  3021377, 0x6387a70a85d46baf}, {  6972593, 0x88f1d2640adb89e1},
                       { 13466917, 0x9fdc1f4092b15d69}, { 20996011, 0x5fc58920a821da11},
                       { 24036583, 0xcbdef38a0bdc4f00}, { 25964951, 0x62eb3ff0a5f6237c},
                       { 30402457, 0x0b8600ef47e69d27}, { 32582657, 0x02751b7fcec76bb1},
                       { 37156667, 0x67ad7646a1fad514}, { 42643801, 0x8f90d78d5007bba7},
                       { 43112609, 0xe86891ebf6cd70c4}, { 57885161, 0x76c27556683cd84d}};

  for(i = 0; i < 20; i++)
  {
    q = test[i].exponent;
    expected_residue = test[i].residue;
    g_fftlen = 0;
    do
    {
      restarting = 0;
      if(!x_packed) x_packed = read_checkpoint(q);
      x_int = init_lucas(x_packed, q, &n, &j, &offset, NULL, NULL, NULL);
      if(!x_int) exit (2);
      if(!restarting) printf ("Starting self test M%d fft length = %dK\n", q, n/1024);
      gettimeofday (&time0, NULL);
      if(g_ro) restarting = round_off_test(q, n, &j, &offset);
      if(restarting) close_lucas (x_int);
      fflush (stdout);

      for (; !restarting && j <= last; j++)
      {
        if(j == last) error_flag = 1;
        else if(j % 100 == 0) error_flag = 2;
        else error_flag = 0;
        terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
        if(error_flag == 2 && j <= 1000) reset_err(&maxerr, 0.75);
        if(terr > 0.4)
        {
          g_fftlen++;
          restarting = 1;
        }
      }
    }
    while (restarting);
    cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
    standardize_digits(x_int, q, n, 0, n);
    residue = find_residue(x_int, q, n, offset);
    gettimeofday (&time1, NULL);
    diff = time1.tv_sec - time0.tv_sec;
    diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
    printf ("Iteration %d ", j - 1);
    printbits(residue, q, n, offset, 0, 0);
    printf (", error = %5.5f, real: ", maxerr);
    print_time_from_seconds ((unsigned) diff, NULL, 0);
    printf (", %4.4f ms/iter\n", diff1 / 1000.0 / last);
    fflush (stdout);

    close_lucas (x_int);
    free ((char *) x_packed);
    x_packed = NULL;
    if(residue != expected_residue)
    {
      printf("Expected residue [%016llx] does not match actual residue [%016llx]\n\n", expected_residue, residue);
      fflush (stdout);
      bad_selftest++;
    }
    else
    {
      printf("This residue is correct.\n\n");
      fflush (stdout);
    }
  }
  if (bad_selftest)
  {
    fprintf(stderr, "Error: There ");
    bad_selftest > 1 ? fprintf(stderr, "were %d bad selftests!\n",bad_selftest) : fprintf(stderr, "was a bad selftest!\n");
  }
}

/**************************************************************************
 *                                                                        *
 *                          Keyboard Interaction                          *
 *                                                                        *
 **************************************************************************/
void
SetQuitting (int sig)
{
  g_qu = 1;
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

int interact(int );
/**************************************************************
 *
 *      Main Function
 *
 **************************************************************/
int
check (int q)
{
  int  *x_int = NULL;
  unsigned *x_packed = NULL;
  int n, j, last = q - 2;
  int offset;
  float maxerr, temperr, terr = 0.0f;
  int interact_result = 0;
  timeval time0, time1;
  unsigned long long total_time = 0, diff, diff1, diff2;
  unsigned long long time_adj = 0;
  unsigned long long residue;
  unsigned iter_adj = 0;
  int last_chk = 0;
  int restarting;
  int retry;
  int error_type = -1;
  int fft_reset = 0;
  int error_flag;
  int error_reset = 1;
  //char version[80] = {0};
  //unsigned int temp = 0;
  //unsigned int speed = 0;
  //nvmlDevice_t device1;
  //nvmlUtilization_t utilization;

  //nvmlInit();
  //nvmlDeviceGetHandleByIndex (0, &device1);
  //nvmlSystemGetDriverVersion (version, 80);

  signal (SIGTERM, SetQuitting);
  signal (SIGINT, SetQuitting);
  do
  {				/* while (restarting) */
    maxerr = temperr = 0.25;

    if(!x_packed) x_packed = read_checkpoint(q);
    x_int = init_lucas(x_packed, q, &n, &j, &offset, &total_time, &time_adj, &iter_adj);
    if(!x_int) exit (2);

    restarting = 0;
    if(j == 1)
    {
      printf ("Starting M%d fft length = %dK\n", q, n/1024);
      if(g_ro)
      {
        restarting = round_off_test(q, n, &j, &offset);
        iter_adj = 1000;
        if(!restarting)
        {
          cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
          standardize_digits(x_int, q, n, 0, n);
          residue = find_residue(x_int, q, n, offset);
          set_checkpoint_data(x_packed, q, j, offset, total_time, time_adj, iter_adj);
          pack_bits(x_int, x_packed, q, n);
          write_checkpoint(x_packed, q, residue);
          last_chk = j;
        }
      }
    }
    else
    {
      printf ("\nContinuing M%d @ iteration %d with fft length %dK, %5.2f%% done\n\n", q, j, n/1024, (float) j/q*100);
      last_chk = j;
    }
    fflush (stdout);
    if(!restarting)
    {
      gettimeofday (&time0, NULL);
    }
    for (; !restarting && j <= last; j++) // Main LL loop
    {
	    error_flag = 0;
	    if (j % g_cpi == 0 || j == last) error_flag = 1;
      else if ((j % 100) == 0) error_flag = 2;
      terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
      if(terr < 0.0)  //cufft timeout error
      {
        printf("Resetting device and restarting from last checkpoint.\n\n");
        cudaDeviceReset();
        init_device(g_dn, 0);
        restarting = 1;
      }
      else
      {
        if ((error_flag & 1) || g_qu) //checkpoint iteration or guitting
        {
          if (!(error_flag & 1)) //quitting, but g_int not up to date, do 1 more iteration
          {
            j++;
            error_flag = 1;
            terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
          }
          if(terr <= 0.4)
          {
            cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
            standardize_digits(x_int, q, n, 0, n);
            residue = find_residue(x_int, q, n, offset);
            gettimeofday (&time1, NULL);
            diff = time1.tv_sec - time0.tv_sec;
            diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
            total_time += diff1;
            diff2 = (total_time - time_adj) * (last - j) / (j - iter_adj) / 1000000;
            time0.tv_sec = time1.tv_sec;
            time0.tv_usec = time1.tv_usec;
            set_checkpoint_data(x_packed, q, j + 1, offset, total_time, time_adj, iter_adj);
            pack_bits(x_int, x_packed, q, n);
            write_checkpoint(x_packed, q, residue);
            if(g_qu)
            {
              printf(" Estimated time spent so far: ");
              print_time_from_seconds((unsigned) (total_time / 1000000), NULL,0);
              printf("\n\n");
              j = last + 1;
            }
            else if(j != last) //screen output
            {
              if(g_output_interval) process_output(q, n, j, offset, last_chk, maxerr, residue, diff, diff1, diff2, total_time);
              else
              {
                printf ("Iteration %d ", j);
                printbits(residue, q, n, offset, 0, 0);
                printf (", error = %5.5f, real: ", maxerr);
                print_time_from_seconds ((unsigned) diff, NULL, 0);
                printf (", %4.4f ms/iter, ETA: ", diff1 / 1000.0 / (j - last_chk));
                print_time_from_seconds ((unsigned) diff2, NULL, 0);
                printf (", %5.2f%%\n", (float) j/q*100);
              }
   //nvmlDeviceGetTemperature (device1,NVML_TEMPERATURE_GPU, &temp);
  //nvmlDeviceGetFanSpeed (device1, &speed);
 // nvmlDeviceGetUtilizationRates (device1, &utilization);
 //printf("%s, %u %u %u %u\n", version, temp, speed, utilization.gpu, utilization.memory);
             fflush (stdout);
              last_chk = j;
              if(!error_reset) reset_err(&maxerr, g_er / 100.0f); // Instead of tracking maxerr over whole run, reset it at each checkpoint.
              if(fft_reset) // Larger fft fixed the error, reset fft and continue
              {
                g_fftlen--;
                restarting = 1;
                printf("Resettng fft.\n\n");
                fft_reset = 0;
                retry = 1;
              }
              if(error_flag & 4) // 64 bit carries fixed the error, resume with 32 bit carries
              {
                printf("Resuming with 32 bit carries.\n\n");
                error_flag -= 4;
                retry = 0;
              }
               if(retry == 1) // Retrying fixed the error
              {
                printf("Looks like the error went away, continuing.\n\n");
                retry = 0;
              }
            }
          }
        }
        if (terr > 0.4)
        {
          if(terr > 0.60)
          {
            printf("Overflow error at iteration = %d, fft = %dK\n", j, n / 1024);
            error_type = 0;
          }
          else
          {
            printf ("Round off error at iteration = %d, err = %0.5g > 0.40, fft = %dK.\n", j,  terr, n / 1024);
            error_type = 1;
          }
          if(g_fftlen >= g_ubi) // fft is at upper bound. overflow errors possible
          {
            printf("Decreasing fft and restarting from last checkpoint.\n\n");
            g_fftlen--;
          }
          else if(g_fftlen <= g_lbi) // fft is at lower bound, round off errors possible
          {
            printf("Increasing fft and restarting from last checkpoint.\n\n");
            g_fftlen++;
          }
          else // fft is not a boundary case
          {
            if(retry)  // Redo from last checkpoint, the more complete rounding/balancing
            {          // on cpu sometimes takes care of the error. Hardware errors go away too.
              printf("Restarting from last checkpoint to see if the error is repeatable.\n\n");
              retry = 0;
            }
            else // Retrying didn't fix the error
            {
              if(error_type) //Round off error
              {
                if(fft_reset) // Larger fft didn't fix the error, give up
                {
                  printf("The error won't go away. I give up.\n\n");
                  exit(0);
                }
                else // Larger fft will usually fix a round off error
                {
                  printf("The error persists.\n");
                  printf("Trying a larger fft until the next checkpoint.\n\n");
                  g_fftlen++;
                  fft_reset = 1;
                }
              }
              else// Overflow error
              {
                if(error_flag & 4) //64 bit carries didn't fix the error
                {
                  if(fft_reset) // Larger fft didn't fix the error, give up
                  {
                    printf("The error won't go away. I give up.\n\n");
                    exit(0);
                  }
                  else // Try larger fft
                  {
                    printf("The error still persists.\n");
                    printf("Trying a larger fft until the next checkpoint.\n\n");
                    g_fftlen++;
                    fft_reset = 1;
                  }
                }
                else // Try 64 bit carries
                {
                  printf("The error persists.\n");
                  printf("Trying again from the last checkpoint with 64 bit carries.\n\n");
                  error_flag |= 4;
                }
              }
            }
          }
          restarting = 1;
          reset_err(&maxerr, 0.25);
          error_reset = 1;
        }
	      else if(error_reset && (error_flag & 2))
        {
            if(terr < temperr + 0.0001f)
            {
              maxerr *= 0.5f;
              temperr = maxerr;
            }
            else error_reset = 0;
        }

        if ( g_ki && !restarting && !g_qu && (!(j & 15)) && _kbhit()) interact_result = interact(n);
        if(interact_result & 3)
        {
          if(!(error_flag & 1))
          {
            j++;
            error_flag |= 1;
            terr = lucas_square (q, n, j, last, &offset, &maxerr, error_flag);
            if(terr <= 0.40)
            {
              cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
              standardize_digits(x_int, q, n, 0, n);
              gettimeofday (&time1, NULL);
              diff = time1.tv_sec - time0.tv_sec;
              diff1 = 1000000 * diff + time1.tv_usec - time0.tv_usec;
              total_time += diff1;
              time_adj = total_time;
              iter_adj = j + 1;
              set_checkpoint_data(x_packed, q, j + 1, offset, total_time, time_adj, iter_adj);
              if(interact_result & 1)
              {
                pack_bits(x_int, x_packed, q, n);
                reset_err(&maxerr, 0.0);
                error_reset = 1;
              }
            }
          }
          if(interact_result & 1) restarting = 1;
        }
      }
	    interact_result = 0;
	    fflush (stdout);
	  } /* end main LL for-loop */

    if (!restarting && !g_qu)
	  { // done with test
	    gettimeofday (&time1, NULL);
	    FILE* fp = fopen_and_lock(g_RESULTSFILE, "a");
	    if(!fp)
	    {
	      fprintf (stderr, "Cannot write results to %s\n\n", g_RESULTSFILE);
	      exit (1);
	    }
	    cutilSafeCall (cudaMemcpy (x_int, g_xint, sizeof (int) * n, cudaMemcpyDeviceToHost));
      if (standardize_digits(x_int, q, n, 0, n))
      {
        printf ("M( %d )P, n = %dK, %s", q, n / 1024, program);
        if (fp) fprintf (fp, "M( %d )P, n = %dK, %s", q, n / 1024, program);
      }
	    else
      {
        residue = find_residue(x_int, q, n, offset);
        printbits(residue, q, n, offset, fp, 1);
      }
      diff = time1.tv_sec - time0.tv_sec;
      total_time +=  diff * 1000000 + time1.tv_usec - time0.tv_usec;
      printf (", estimated total time = ");
      print_time_from_seconds((unsigned) (total_time / 1000000), NULL, 0);

	    if( g_AID[0] && strncasecmp(g_AID, "N/A", 3) )
      { // If (AID is not empty), AND (AID is NOT "N/A") (case insensitive)
        fprintf(fp, ", g_AID: %s\n", g_AID);
	    }
      else fprintf(fp, "\n");
	    unlock_and_fclose(fp);
	    fflush (stdout);
	    rm_checkpoint (q);
      g_fft_count = 0;
	    printf("\n\n");
	  }
    if(terr < 0.0) free_host(x_int);
    else close_lucas(x_int);
  }
  while (restarting);
  free ((char *) x_packed);
  x_packed = NULL;
  //nvmlShutdown();
  return (0);
}


void parse_args(int argc, char *argv[], int* q, int* cufftbench_s, int* cufftbench_e, int* cufftbench_d);
		/* The rest of the opts are global */

void encode_output_options(void)
{
  int i = 0, j = 0, temp;
  unsigned k;
  char token[196];
  char c;

  c = g_output_string[0];
  while(c)
  {
    if(c == '%')
    {
      i++;
      c = g_output_string[i];
      switch (c)
      {
        case 'a' : //day of week
                   g_output_code[j] = 0;
                   j++;
                   break;
        case 'M' : //month
                   g_output_code[j] = 1;
                   j++;
                   break;
        case 'D' : //date
                   g_output_code[j] = 2;
                   j++;
                   break;
        case 'h' : //hour
                   g_output_code[j] = 3;
                   j++;
                   break;
        case 'm' : //minutes
                   g_output_code[j] = 4;
                   j++;
                   break;
        case 's' : //seconds
                   g_output_code[j] = 5;
                   j++;
                   break;
        case 'y' : //year
                   g_output_code[j] = 6;
                   j++;
                   break;
        case 'z' : //time zone
                   g_output_code[j] = 7;
                   j++;
                   break;
        case 'p' : //testing exponent
                   g_output_code[j] = 8;
                   j++;
                   break;
        case 'P' : //testing Mersenne
                   g_output_code[j] = 9;
                   j++;
                   break;
        case 'f' : //fft as a multiple of K
                   g_output_code[j] = 10;
                   j++;
                   break;
        case 'F' : //fft
                   g_output_code[j] = 11;
                   j++;
                   break;
        case 'i' : //iteration number
                   g_output_code[j] = 12;
                   j++;
                   break;
        case 'o' : //offset
                   g_output_code[j] = 13;
                   j++;
                   break;
        case 'r' : //residue
                   g_output_code[j] = 14;
                   j++;
                   break;
        case 'R' : //residue with upper case hex
                   g_output_code[j] = 15;
                   j++;
                   break;
        case 'C' : //time since last checkpoint hh:mm:ss
                   g_output_code[j] = 16;
                   j++;
                   break;
        case 'T' : //eta d:hh:mm:ss
                   g_output_code[j] = 17;
                   j++;
                   break;
        case 'N' : //residue with upper case hex
                   g_output_code[j] = 18;
                   j++;
                   break;
        case '%' : //residue with upper case hex
                   g_output_code[j] = 19;
                   j++;
                   break;
        case 'x' : //round off error
                   g_output_code[j] = 25;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
        case 'q' : //iteration timing
                   g_output_code[j] = 26;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                  j++;
                   break;
        case 'c' : //checkpoint interval timing ms/iter
                   g_output_code[j] = 27;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
        case 'Q' : //checkpoint interval timing iter/sec
                   g_output_code[j] = 28;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
        case 'd' : //percent completed
                   g_output_code[j] = 29;
                   j++;
                   i++;
                   temp = g_output_string[i] - 48;
                   if(temp > 5 || temp < 0) temp = 5;
                   g_output_code[j] = 5 - temp + (temp == 0);
                   j++;
                   break;
       default  :
                   break;
      }//end switch
      i++;
      c = g_output_string[i];
    }
    else // c != '%', write the string up to the next % to array
    {
      k = 0;
      while(c && c != '%')
      {
        token[k] = c;
        k++;
        i++;
        c = g_output_string[i];
      }
      token[k] = '\0';
      int m = 30, found = 0;
      while(m < 50 && !found)
      {
        if (strlen(g_output[m]) == 0)
        {
          k = (k < 32 ? k : 32);
          strncpy(g_output[m], token, k);
          g_output_code[j] = m;
          j++;
          found = 1;
        }
        else if(!strncmp(token, g_output[m], k) && strlen(g_output[m]) == k)
        {
          g_output_code[j] = m;
          j++;
          found = 1;
       }
        m++;
      }
    }
  }//end while(c)
  g_output_code[j] = -1;
}

void process_options(void)
{
#define THREADS_DFLT 256
#define CHECKPOINT_ITER_DFLT 10000
#define SAVE_FOLDER_DFLT "savefiles"
#define S_F_DFLT 0
#define K_F_DFLT 0
#define D_F_DFLT 0
#define ROT_F_DFLT 0
#define POLITE_DFLT 1
#define WORKFILE_DFLT "worktodo.txt"
#define RESULTSFILE_DFLT "results.txt"
#define ER_DFLT 85
#define OUTPUT_DFLT "0"
#define HEADER_DFLT "0"
#define HEADERINT_DFLT 15

  char fft_str[192] = "\0";


  if (file_exists(g_INIFILE))
  {
    if( g_sf < 0 &&           !IniGetInt(g_INIFILE, "SaveAllCheckpoints", &g_sf, S_F_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveAllCheckpoints; using default: off\n")*/;
    if( g_sf > 0 &&           !IniGetStr(g_INIFILE, "SaveFolder", g_folder, SAVE_FOLDER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option SaveFolder; using default: \"%s\"\n", SAVE_FOLDER_DFLT)*/;
    if( g_ki < 0 &&           !IniGetInt(g_INIFILE, "Interactive", &g_ki, 0) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Interactive; using default: off\n")*/;
    if( g_ro < 0 &&           !IniGetInt(g_INIFILE, "RoundoffTest", &g_ro, ROT_F_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option RoundoffTest; using default: off\n")*/;
    if( g_df < 0 &&           !IniGetInt(g_INIFILE, "PrintDeviceInfo", &g_df, D_F_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option PrintDeviceInfo; using default: off\n")*/;
    if( g_po < 0 &&	          !IniGetInt(g_INIFILE, "Polite", &g_po, POLITE_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Polite; using default: %d\n", POLITE_DFLT)*/;
    if( g_dn < 0 &&           !IniGetInt(g_INIFILE, "DeviceNumber", &g_dn, 0) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option DeviceNumber; using default: 0\n")*/;
    if( g_cpi < 1 &&          !IniGetInt(g_INIFILE, "CheckpointIterations", &g_cpi, CHECKPOINT_ITER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option CheckpointIterations; using default: %d\n", CHECKPOINT_ITER_DFLT)*/;
    if( g_er < 0 &&           !IniGetInt(g_INIFILE, "ErrorReset", &g_er, ER_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option ErrorReset using default: 85\n")*/;
    if( g_th < 0 &&        !IniGetInt3(g_INIFILE, "Threads", &g_thr[0], &g_thr[1], &g_thr[2], THREADS_DFLT) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option Threads; using default: 256 128 128%d\n", THREADS_DFLT)*/;
    if( g_fftlen < 0 &&       !IniGetStr(g_INIFILE, "FFTLength", fft_str, "\0") )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option FFTLength; using autoselect.\n")*/;
    if( !g_input_file[0] &&	  !IniGetStr(g_INIFILE, "WorkFile", g_input_file, WORKFILE_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option WorkFile; using default \"%s\"\n", WORKFILE_DFLT);
      /* I've readded the warnings about worktodo and results due to the multiple-instances-in-one-dir feature. */
    if( !g_RESULTSFILE[0] &&  !IniGetStr(g_INIFILE, "ResultsFile", g_RESULTSFILE, RESULTSFILE_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option ResultsFile; using default \"%s\"\n", RESULTSFILE_DFLT);
    if( g_output_string[0] < 0 &&  !IniGetStr(g_INIFILE, "OutputString", g_output_string, OUTPUT_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option OutputString; using default \"%s\"\n", OUTPUT_DFLT);
    if( g_output_header[0] < 0 &&  !IniGetStr(g_INIFILE, "OutputHeader", g_output_header, HEADER_DFLT) )
      fprintf(stderr, "Warning: Couldn't parse ini file option OutputHeader; using default \"%s\"\n", HEADER_DFLT);
    if( g_output_interval < 0 &&  !IniGetInt(g_INIFILE, "OutputHInterval", &g_output_interval, 0) )
      /*fprintf(stderr, "Warning: Couldn't parse ini file option OutputHInterval; using default \"%s\"\n", HEADERINT_DFLT)*/;
  }
  else // no ini file, set default values
  {
    fprintf(stderr, "Warning: Couldn't find .ini file. Using defaults for non-specified options.\n");
    if( g_cpi < 1 ) g_cpi = CHECKPOINT_ITER_DFLT;
    if( g_thr[0] < 1 ) g_thr[0] = THREADS_DFLT;
    if( g_fftlen < 0 ) g_fftlen = 0;
    if( g_sf < 0 ) g_sf = S_F_DFLT;
    if( g_ki < 0 ) g_ki = K_F_DFLT;
    if( g_dn < 0 ) g_dn = 0;
    if( g_df < 0 ) g_df = D_F_DFLT;
    if( g_po < 0 ) g_po = POLITE_DFLT;
    if( g_output_string[0] < 0) sprintf(g_output_string, OUTPUT_DFLT);
    if( !g_input_file[0] ) sprintf(g_input_file, WORKFILE_DFLT);
    if( !g_RESULTSFILE[0] ) sprintf(g_RESULTSFILE, RESULTSFILE_DFLT);
    if( !g_output_header[0] ) sprintf(g_output_header, HEADER_DFLT);
    if( !g_output_string[0] ) sprintf(g_output_string, OUTPUT_DFLT);
    if( g_output_interval < 0 ) g_output_interval = HEADERINT_DFLT;
  }
  if( g_fftlen < 0 ) g_fftlen = fft_from_str(fft_str); // possible if -f not on command line
  if (g_po == 0)
  {
    g_pf = 0;
    g_po = 1;
  }
  else g_pf = 1;
  encode_output_options();

}

void make_savefile_folder(void)
{
#ifndef _MSC_VER
  mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  if (mkdir (g_folder, mode) != 0) fprintf (stderr, "mkdir: cannot create directory `%s': File exists\n", g_folder);
#else
  if (_mkdir (g_folder) != 0) fprintf (stderr, "mkdir: cannot create directory `%s': File exists\n", g_folder);
#endif
}

int main (int argc, char *argv[])
{
  /* "Production" opts to be read in from command line or ini file */
  int q = -1;
  int f_f;
  int cufftbench_s, cufftbench_e, cufftbench_d;

  printf("\n");

  cufftbench_s = cufftbench_e = cufftbench_d = g_rt = 0;
  g_cpi = g_er = g_thr[0] = g_fftlen = g_pf = g_po = g_sf = g_df = g_ki = g_ro = g_th = -1;
  g_output_string[0] = g_output_header[0] = g_output_interval = -1;
  g_AID[0] = g_input_file[0] = g_RESULTSFILE[0] = 0; /* First character is null terminator */

  parse_args(argc, argv, &q, &cufftbench_s, &cufftbench_e, &cufftbench_d); // The rest of the args are globals
  process_options();
  f_f = g_fftlen; // if the user has given an override... then note this length must be kept between tests
  init_device (g_dn, g_df);
  if (g_rt) check_residue ();
  else if (cufftbench_d > 0) cufftbench (cufftbench_s, cufftbench_e, cufftbench_d, g_dn);
  else if (cufftbench_e > 0) memtest (cufftbench_s, cufftbench_e, g_dn);
  else
  {
    if (g_sf) make_savefile_folder();
    if (q <= 0)
    {
      do
      {
        g_fftlen = f_f; // fftlen and AID change between tests, so be sure to reset them
        g_AID[0] = 0;
        if(get_next_assignment(g_input_file, &q, &g_fftlen, &g_AID) > 0) exit (2);
        check (q);
        if(!g_qu && clear_assignment(g_input_file, q)) exit (2);
      } while(!g_qu);
    }
    else
    {
      if (!valid_assignment(q, g_fftlen)) exit (2); //! v_a prints warning
      check (q);
    }
  }
}


void parse_args(int argc,
                char *argv[],
                int* q,
                int* cufftbench_s,
                int* cufftbench_e,
                int* cufftbench_d)
{

  while (argc > 1)
  {
    if (strcmp (argv[1], "-h") == 0) //Help
    {
  	  fprintf (stderr, "$ CUDALucas -h|-v\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-c checkpoint_iteration] [-f fft_length] [-s folder] [-polite iteration] [-k] exponent|input_filename\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] [-info] [-i inifile] [-threads 32|64|128|256|512|1024] [-polite iteration] -r\n\n");
  	  fprintf (stderr, "$ CUDALucas [-d device_number] [-info] -cufftbench start end passes\n\n");
      fprintf (stderr, "                       -h          print this help message\n");
      fprintf (stderr, "                       -v          print version number\n");
      fprintf (stderr, "                       -info       print device information\n");
      fprintf (stderr, "                       -i          set .ini file name (default = \"CUDALucas.ini\")\n");
  	  fprintf (stderr, "                       -threads    set threads numbers (eg -threads 256 128 128)\n");
  	  fprintf (stderr, "                       -f          set fft length (if round off error then exit)\n");
  	  fprintf (stderr, "                       -s          save all checkpoint files\n");
  	  fprintf (stderr, "                       -polite     GPU is polite every n iterations (default -polite 0) (-polite 0 = GPU aggressive)\n");
  	  fprintf (stderr, "                       -cufftbench exec CUFFT benchmark. Example: $ ./CUDALucas -d 1 -cufftbench 1 8192 2\n");
 	    fprintf (stderr, "                       -r          exec residue test.\n");
  	  fprintf (stderr, "                       -k          enable keys (see CUDALucas.ini for details.)\n\n");
  	  exit (2);
    }
    else if (strcmp (argv[1], "-v") == 0) //Version
    {
      printf("%s\n\n", program);
      exit (2);
    }
    else if (strcmp (argv[1], "-polite") == 0) // Polite option
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -polite option\n\n");
	      exit (2);
	    }
	    g_po = atoi (argv[2]);
	    if (g_po == 0)
	    {
	      g_pf = 0;
	      g_po = 1;
	    }
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-r") == 0) // Residue check
	  {
	    g_rt = 1;
	    argv++;
	    argc--;
	  }
    else if (strcmp (argv[1], "-k") == 0) // Interactive option
	  {
	    g_ki = 1;
	    argv++;
	    argc--;
	  }
    else if (strcmp (argv[1], "-d") == 0) // Device number
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -d option\n\n");
	      exit (2);
	    }
	    g_dn = atoi (argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-i") == 0) //ini file
	  {
	    if(argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -i option\n\n");
	      exit (2);
	    }
	    sprintf (g_INIFILE, "%s", argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-info") == 0) // Print device info
    {
      g_df = 1;
      argv++;
      argc--;
    }
    else if (strcmp (argv[1], "-cufftbench") == 0) //cufftbench parameters
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
    else if (strcmp (argv[1], "-memtest") == 0)// memtest parameters
	  {
	    if (argc < 4 || argv[2][0] == '-' || argv[3][0] == '-' )
	    {
	      fprintf (stderr, "can't parse -memtest option\n\n");
	      exit (2);
	    }
	    *cufftbench_s = atoi (argv[2]);
	    *cufftbench_e = atoi (argv[3]);
	    argv += 3;
	    argc -= 3;
	  }
    else if (strcmp (argv[1], "-threads") == 0) // Threads
	  {
	    if (argc < 5 || argv[2][0] == '-' || argv[3][0] == '-' || argv[4][0] == '-')
	    {
	      fprintf (stderr, "can't parse -threads option\n\n");
	      exit (2);
	    }
	    g_th = 0;
      g_thr[0] = atoi (argv[2]);
	    g_thr[1] = atoi (argv[3]);
	    g_thr[2] = atoi (argv[4]);
	    argv += 4;
	    argc -= 4;
	  }
    else if (strcmp (argv[1], "-c") == 0) // checkpoint iteration
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	    g_cpi = atoi (argv[2]);
	    if (g_cpi == 0)
	    {
	      fprintf (stderr, "can't parse -c option\n\n");
	      exit (2);
	    }
	    argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-f") == 0) //fft length
	  {
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -f option\n\n");
	      exit (2);
	    }
	    g_fftlen = fft_from_str(argv[2]);
      argv += 2;
	    argc -= 2;
	  }
    else if (strcmp (argv[1], "-s") == 0)
	  {
	    g_sf = 1;
	    if (argc < 3 || argv[2][0] == '-')
	    {
	      fprintf (stderr, "can't parse -s option\n\n");
	      exit (2);
	    }
	    sprintf (g_folder, "%s", argv[2]);
	    argv += 2;
	    argc -= 2;
	  }
    else // the prime exponent
    {
      if (*q != -1 || strcmp (g_input_file, "") != 0 )
      {
        fprintf (stderr, "can't parse options\n\n");
        exit (2);
      }
      int derp = atoi (argv[1]);
      if (derp == 0)
      {
        sprintf (g_input_file, "%s", argv[1]);
      }
      else
      {
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
  int max_threads = (int) g_dev.maxThreadsPerBlock;

  switch( c )
  {
    case 'p' :
                g_pf = 1 - g_pf;
                printf ("   -polite %d\n", g_pf * g_po);
                break;
    case 's' :
                if (g_sf == 1)
                {
                  g_sf = 2;
                  printf ("   disabling -s\n");
                }
                else if (g_sf == 2)
                {
                  g_sf = 1;
                  printf ("   enabling -s\n");
                }
                break;
    case 'F' :
                printf(" -- Increasing fft length.\n");
                g_fftlen++;
                return 1;
    case 'f' :
                printf(" -- Decreasing fft length.\n");
                g_fftlen--;
                return 1;
    case 'Q' :
                cutilSafeThreadSync();
                if(g_thr[0] < max_threads && (n % (4 * g_thr[0]) == 0)) g_thr[0] *= 2;
                l = 0;
                k = 32;
                while(k < g_thr[0])
                {
                  l += n / (2 * k);
                  k *= 2;
                }
                g_ttp1 = &g_x[n / 4 * 9 + l];
                printf(" -- norm1 threads increased to %d.\n", g_thr[0]);
                break;
    case 'q' :
                cutilSafeThreadSync();
                if(g_thr[0] > 32 && (n / g_thr[0] <= g_dev.maxGridSize[0])) g_thr[0] /= 2;
                l = 0;
                k = 32;
                while(k < g_thr[0])
                {
                  l += n / (2 * k);
                  k *= 2;
                }
                g_ttp1 = &g_x[n / 4 * 9 + l];
                printf(" -- norm1 threads decreased to %d.\n", g_thr[0]);
                break;
    case 'W' :
                if(g_thr[1] < max_threads&& (n % (8 * g_thr[1]) == 0)) g_thr[1] *= 2;
                printf(" -- mult threads increased to %d.\n", g_thr[1]);
                break;
    case 'w' :
                if(g_thr[1] > 32 && (n / ( 2 * g_thr[1]) <= g_dev.maxGridSize[0])) g_thr[1] /= 2;
                printf(" -- mult threads decreased to %d.\n", g_thr[1]);
                break;
    case 'E' :
                if(g_thr[2] < max_threads) g_thr[2] *= 2;
                printf(" -- norm2 threads increased to %d.\n", g_thr[2]);
                break;
    case 'e' :
                if(g_thr[2] > 32) g_thr[2] /= 2;
                printf(" -- norm2 threads decreased to %d.\n", g_thr[2]);
                break;
    case 'R' :
                if(g_er < 100) g_er += 5;
                printf(" -- error_reset increased to %d.\n", g_er);
                break;
    case 'r' :
                if(g_er > 0) g_er -= 5;
                printf(" -- error_reset decreased to %d.\n", g_er);
                break;
    case 'T' :
                if(g_cpi == 1) g_cpi = 2;
                else if(g_cpi == 2) g_cpi = 5;
                else
                {
                  k = g_cpi;
                  while(k % 10 == 0) k /= 10;
                  if(k == 1) g_cpi *= 2.5;
                  else g_cpi *= 2;
                }
                printf(" -- checkpoint_iter increased to %d.\n", g_cpi);
                break;
    case 't' :
                k = g_cpi;
                if(g_cpi == 5) g_cpi = 2;
                else
                {
                  while (k % 10 == 0) k /= 10;
                  if (k == 25) g_cpi /= 2.5;
                  else if (g_cpi > 1) g_cpi /= 2;
                }
                printf(" -- checkpoint_iter decreased to %d.\n", g_cpi);
                break;
    case 'o' :
                IniGetStr(g_INIFILE, "OutputString", g_output_string, "0");
                IniGetStr(g_INIFILE, "OutputHeader", g_output_header, "0");
                IniGetInt(g_INIFILE, "OutputHInterval", &g_output_interval, 0);
                encode_output_options();
                printf(" -- refreshing output format options.\n");
                break;
    case 'n' :
                printf(" -- resetting timer.\n");
                return 2;
    default  :
                break;
  }
  fflush (stdin);
  return 0;
}
