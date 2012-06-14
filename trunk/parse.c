/*
This file used to be part of mfaktc.
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
This file has been written by Luigi Morelli (L.Morelli@mclink.it) *1

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                
You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/



/*
*1 Luigi initially wrote the two functions get_next_assignment() and
clear_assignment() after we (Luigi and myself (Oliver)) have discussed the
interface. Luigi was so nice to write those functions so I had time to focus
on other parts, this made early (mfaktc 0.07) worktodo parsing posible.
For mfakc 0.15 I've completly rewritten those two functions. The rewritten
functions should be more robust against malformed input. Grab the sources of
mfaktc 0.07-0.14 to see Luigis code.
*/

//Christenson, 2011, needed to expand the functionality, and strongly
// disliked not running everything through the same parse function.
//  after a first re-write with a do-everything parse function that was
//  too complicated for its own good, a second re-write centers around
//  a new function, parse_line, that takes in a line of worktodo and returns
//  a structured result.
// the value of this is that it can be re-targeted to some other forms of
// numbers besides Mersenne exponents.....

/* Bill stole it and modified it for CUDALucas' purposes... :) */

/************************************************************************************************************
 * Input/output file function library                                                                       *
 *                                                   						            *
 *   return codes:											    *
 *     0 - OK												    *
 *     1 - get_next_assignment : cannot open file				                            *
 *     2 - get_next_assignment : no valid assignment found		                        	    *
 *     3 - clear_assignment    : cannot open file <filename>		                    		    *
 *     4 - clear_assignment    : cannot open file "__worktodo__.tmp"					    *
 *     5 - clear_assignment    : assignment not found							    *
 *     6 - clear_assignment    : cannot rename temporary workfile to regular workfile			    *
 ************************************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 131

#ifdef linux /* See next comment */
  #define _fopen fopen
  #define strcopy strncpy
  #define _sprintf sprintf
#else
  #define strncasecmp _strnicmp
  
  /* Everything from here to the next include is to make MSVS happy. */
  #define sscanf sscanf_s /* This only works for scanning numbers, or strings with a defined length (e.g. "%131s") */

  void strcopy(char* dest, char* src, size_t n) 
  {
    strncpy_s(dest, MAX_LINE_LENGTH+1, src, n);
  }

  FILE* _fopen(const char* path, const char* mode) 
  {
    FILE* stream;
    errno_t err = fopen_s(&stream, path, mode);
    if(err) return NULL;
    else return stream;
  }
  
  void _sprintf(char* buf, char* frmt, char* string)
  {	  
	  sprintf_s(buf, MAX_LINE_LENGTH+1, frmt, string);
  }

  #include <winsock2.h>
  int gettimeofday(struct timeval *tv, struct timezone *unused)
  /*
  This is based on a code sniplet from Kevin (kjaget on www.mersenneforum.org)

  This doesn't act like a real gettimeofday(). It has a wrong offset but this is
  OK since CUDALucas only uses this to measure the time difference between two calls
  of gettimeofday().
  */
  {
    static LARGE_INTEGER frequency;
    static int frequency_flag = 0;

    if(!frequency_flag)
    {
      QueryPerformanceFrequency(&frequency);
      frequency_flag = 1;
    }

    if(tv)
    {
      LARGE_INTEGER counter;
      QueryPerformanceCounter(&counter);
      tv->tv_sec =  (long) (counter.QuadPart / frequency.QuadPart);
      tv->tv_usec = (long)((counter.QuadPart % frequency.QuadPart) / ((double)frequency.QuadPart / 1000000.0));
    }
    return 0;
  }
#endif

/***********************************************************************************************************/

int file_exists(char* name) {
  if(name && name[0]) { /* Check for null string */
	FILE* stream;
	if((stream = _fopen(name, "r")))
	  {  
	    fclose(stream);
	    return 1;
	  }
	else return 0;
  } else return 0;
}

int isprime(unsigned int n)
/*
returns
0 if n is composite
1 if n is prime
*/
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

int valid_assignment(int exp, int fftlen)
/*
returns 1 if the assignment is within the supported bounds of CUDALucas,
0 otherwise.
*/
{
  int ret = 1;
  
	// Perhaps add a largest exponent?  
	if(exp < 86243)       {ret = 0; fprintf(stderr, "Warning: exponents < 86243 are not supported!\n");}
	if(!isprime(exp))     {ret = 0; fprintf(stderr, "Warning: exponent is not prime!\n");}
	if(fftlen % (128*32)) {ret = 0; fprintf(stderr, "Warning: FFT length is invalid. See CUDALucas.ini for details about valid lengths.\n");}
	// This doesn't guarantee that it *is* valid, but it will catch horribly bad lengths. 
	// (To do more checking, we'd need access the "threads" variable from CUDALucas.cu.)
	
  return ret;
}

enum ASSIGNMENT_ERRORS
{	NEVER_ASSIGNED=-1,
	OK=0,
	CANT_OPEN_FILE=1,
	VALID_ASSIGNMENT_NOT_FOUND=2,
	CANT_OPEN_WORKFILE=3,
	CANT_OPEN_TEMPFILE=4,
	ASSIGNMENT_NOT_FOUND=5,
	CANT_RENAME =6
};

typedef char LINE_BUFFER[MAX_LINE_LENGTH+1];

enum PARSE_WARNINGS
{
  NO_WARNING=0,
  END_OF_FILE,
  LONG_LINE,
  NO_TEST_EQUAL,
  INVALID_FORMAT,
  INVALID_DATA,
  BLANK_LINE,
  NONBLANK_LINE
};

struct ASSIGNMENT
{
	int exponent;
	int fft_length;
	char hex_key[MAX_LINE_LENGTH+1];	// optional assignment key....
	char comment[MAX_LINE_LENGTH+1];	// optional comment -- it followed the assignment on the same line.
};
// Should we include FFT length in this?

// note:  parse_worktodo_line() is a function that
//	returns the text of the line, the assignment data structure, and a success code.
enum PARSE_WARNINGS parse_worktodo_line(FILE *f_in, struct ASSIGNMENT *assignment, LINE_BUFFER *linecopy, char * *endptr)
/*
input
  f_in: an open file from where data is read
output
  assignment: structure of line, with any assignment if found
  linecopy: a copy of the last read line
  endptr: the end of data
*/
{
  #ifdef EBUG
  printf("Entered p_w_l\n");
  #endif
  char line[MAX_LINE_LENGTH+1], *ptr, *ptr_start, *ptr_end, *ptr2;
  
  /* See below about LONG_LINE */
  //int c;	// extended char pulled from stream;

  int scanpos;
  int number_of_commas; // Originally this was unsigned, causing lots of problems in my outer for(), and it took me a while to track down :P
  			// (It didn't help that I was printf'ing '%d' for an unsigned int, thus masking the problem to look correct. :P)
  int comment_on_line = 0; // Set to the delimiting char if the line has a comment following the assignment.

  unsigned long proposed_exponent;
  unsigned long proposed_fftlen;
  assignment->fft_length = 0;
  assignment->exponent = 0;
  assignment->hex_key[0] = 0;

  if(NULL==fgets(line, MAX_LINE_LENGTH+1, f_in))
  {
    return END_OF_FILE;
  }

  if (linecopy != NULL)	{ // maybe it wasn't needed....
    strcopy(*linecopy, line, MAX_LINE_LENGTH+1);	// this is what was read...
    if( NULL != strchr(*linecopy, '\n') )
      *strchr(*linecopy, '\n') = '\0';
  }
  if((strlen(line) == MAX_LINE_LENGTH) && (!feof(f_in)) && (line[strlen(line)-1] !='\n') ) // long lines disallowed,
  {  
    return LONG_LINE;
    // I see no reason to go to all of the following fuss
    
    /*reason = LONG_LINE;
    do
    {
      c = fgetc(f_in);
      if ((EOF == c) ||(iscntrl(c)))	// found end of line
        break;
    }
    while(1); */
  }

  if (linecopy != NULL)
    *endptr = *linecopy;	// by default, non-significant content is whole line
  
  #ifdef EBUG
  printf("Line: %s", line);
  #endif

  ptr=line;
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// skip leading spaces
    ptr++;
  if ('\0' == ptr[0])	// blank line...
    return BLANK_LINE;
  if( ('\\'== ptr[0]) && ('\\'==ptr[1]) ) 
    return NONBLANK_LINE;		// it's a comment, so ignore....don't care about long lines either..
  if( ('/' == ptr[0]) && ('/'==ptr[1]) )
    return NONBLANK_LINE;		// it's a comment, so ignore....don't care about long lines either..
  if( ('#' == ptr[0]) )
    return NONBLANK_LINE;		// it's a comment, so ignore....don't care about long lines either..
  if ( (strncasecmp("Test=", ptr, 5) != 0) &&
       (strncasecmp("DoubleCheck=", ptr, 12) != 0) ) // does the line start with "Test=" or "DoubleCheck="? 
                                                     // (case-insensitive)
    return NO_TEST_EQUAL;

  ptr = 1+ strchr(ptr, '=');	// don't rescan..
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// ignore blanks...
    ptr++;
  number_of_commas = 0;
  for(scanpos = 0; scanpos < strlen(ptr); scanpos++)
  {
    if(ptr[scanpos] == ',')
      number_of_commas++; // count the number of ',' in the line
    if ((ptr[scanpos] == '\\') && (ptr[scanpos+1] == '\\')) {
      comment_on_line = '\\';
      break;	// comment delimiter
    }
    if ((ptr[scanpos] == '/') && (ptr[scanpos+1] == '/')) {
      comment_on_line = '/';
      break;	// //comment delimiter
    }
    if (ptr[scanpos] == '#') {
      comment_on_line = '#';
      break;    // comment delimiter
    }
  }
  #ifdef EBUG
  printf("Scanned %d commas\n", number_of_commas);
  #endif
  if (number_of_commas > 4)	// must have less than 5 commas... (possible fields are key,exp,fft,tf,p-1)
    return INVALID_FORMAT;
    
  for(; number_of_commas >= 0; number_of_commas--) {
    // i is number of commas ahead of ptr (or, there's one more field than commas, so iterate n+1 times)
    while (isspace(ptr[0]))	// ignore blanks...
      ptr++;
    if(number_of_commas > 0) // then there's at least one comma left
      ptr_end = strchr(ptr, ','); // guaranteed that strchr isn't null because of the if
    else if(comment_on_line) // no more commas, but there is a comment
      ptr_end = strchr(ptr, comment_on_line); // (see declaration of c_on_l or lines 285,289,293 for explanation)
    else { // no commas or comments
      ptr_end = strchr(ptr, '\n');
      if(ptr_end == NULL)
        ptr_end = strchr(ptr, '\0');
    }
    #ifdef EBUG
    printf("In main for() loop, %d commas left\n", number_of_commas);
    #endif
    for(ptr_start = ptr; ptr_start < ptr_end; ptr_start++) { 
      
      #ifdef EBUG
      printf("Looping on chars, ptr_start = %c\n", *ptr_start);
      #endif
      if( ('A' <= *ptr_start && *ptr_start <= 'F') || ('a' <= *ptr_start && *ptr_start <= 'f') ) {
      // we have some sort of hex assignment key, or "N/A". Either way, it's an AID of some sort.
        #ifdef EBUG
        printf("Branched on AID, trigger is %c\n", *ptr_start);
        #endif
        strcopy(assignment->hex_key, ptr, 1+(ptr_end - ptr));	// copy the comma..
        *strchr(assignment->hex_key, ',') = '\0';	// null-terminate key
        goto outer_continue;
      }
      else if( *ptr_start == 'k' || *ptr_start == 'K' ) {
      // we've found a fft length field.
        #ifdef EBUG
        printf("Branched on FFT-K\n");
        #endif
        errno = 0;
        proposed_fftlen = strtoul(ptr, &ptr2, 10) * 1024; // Don't forget the K ;)
        if(ptr == ptr2 || ptr2 < ptr_start) // the second condition disallows space between the num and K, so "1444 K" will fail,
          return INVALID_FORMAT;            // but it also disallows "1rrrK", so I think it's worth it.
        if(errno != 0 || proposed_fftlen > INT_MAX)
          return INVALID_DATA;
        assignment->fft_length = proposed_fftlen;
        goto outer_continue;
      }
      else if( *ptr_start == 'm' || *ptr_start == 'M' ) {
      // we've found a fft length field.
        #ifdef EBUG
        printf("Branched on FFT-M\n");
        #endif
        errno = 0;
        proposed_fftlen = strtoul(ptr, &ptr2, 10) * 1024*1024; // Don't forget the M ;)
        if(ptr == ptr2 || ptr2 < ptr_start) // the second condition disallows space between the num and K, so "1444 M" will fail,
          return INVALID_FORMAT;            // but it also disallows "1rrrK", so I think it's worth it.
        if(errno != 0 || proposed_fftlen > INT_MAX)
          return INVALID_DATA;
        assignment->fft_length = proposed_fftlen;
        goto outer_continue;
        
      } else { // Not special, so we must continue checking the rest of the chars in the field
        continue;
      }     
    } // end inner for()
    // Now we know there's nothing special about this field, so
    // we must assume it's the exponent, except to assume that the largest number 
    // read in is the exponent (to filter out TF lim or P-1 bool)
    #ifdef EBUG
    printf("Branched on default\n");
    #endif
    ptr_start = ptr; /* Nothing special, so reset ptr_start to the start */
    if ('M' == *ptr_start)	// M means Mersenne exponent...
      ptr_start++;
    errno = 0;
    proposed_exponent = strtoul(ptr_start, &ptr2, 10);
    if(ptr_start == ptr2)
      return INVALID_FORMAT; // no conversion
    if(errno != 0 || proposed_exponent > INT_MAX)
      return INVALID_DATA;
    #ifdef EBUG
    printf("'Expo' conversion is %ld\n", proposed_exponent);
    #endif
    if( proposed_exponent > assignment->exponent ) // don't clobber larger values, this is our TF/P-1 filter
      assignment->exponent = (int)proposed_exponent;
    
    outer_continue: /* One nice feature Python has is putting "else"s on loops, only to be executed when NOT "break"-ed from.
                 That's exactly what I'm duplicating here with the inner loop and default "expo" branching. */
    ptr = 1 + ptr_end; // Reset for the next field (*ptr == '\n' || *ptr == comment-delimiter when we're done)
  } // end outer for()
  // now we've looped over all fields in worktodo line
  ptr = ptr_end;
  #ifdef EBUG
  printf("Left for()\n");
  #endif
  if(*ptr == '\n' || *ptr == '\0') // no comment (lol)
    assignment->comment[0] = '\0';
  else if(comment_on_line) {
    
    // Don't include delimters in actual comment
    if(*ptr == '/' || *ptr == '\\')
      ptr += 2;
    else if(*ptr == '#')
      ptr++;
    else { fprintf(stderr, "Wow, something's screwed up. Please file as detailed a bug report as possible.\n\n"); exit(7);
         }
      
    while (('\0'!=ptr[0]) && isspace(ptr[0]))	// ignore blanks...
      ptr++;
    if (*ptr != '\0') {
      if (NULL != strchr(ptr,'\n'))		// kill off any trailing newlines...
        *strchr(ptr,'\n') = '\0';
      strcopy(assignment->comment, ptr, 1+strchr(ptr,'\0')-ptr);
    } else assignment->comment[0] = '\0';
  
  } else /* No comment on line, but no terminating null or newline? */ 
    { fprintf(stderr, "Wow, something's way screwed up. Please file as detailed a bug report as possible.\n\n"); exit(77);
    }

  if (linecopy != NULL)
    *endptr = *linecopy + (ptr_end - line);
  
  if(assignment->exponent < 100 || (0 < assignment->fft_length && assignment->fft_length < 4096) ) // sanity check
    return INVALID_DATA;

  return NO_WARNING;
}

/************************************************************************************************************
 * Function name : get_next_assignment                                                                      *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		int *exponent										    *
 *		int* fft_length									    	    *
 *		char *hex_key[132];								    	    *
 *     OUTPUT :                                        							    *
 *                                                                                                          *
 *     0 - OK												    *
 *     1 - get_next_assignment : cannot open file							    *
 *     2 - get_next_assignment : no valid assignment found						    *
 ************************************************************************************************************/
enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, int *exponent, int* fft_length, LINE_BUFFER *key)
{
  #ifdef EBUG
  printf("Starting GNA.\n");
  #endif
  FILE *f_in;
  enum PARSE_WARNINGS value;
  struct ASSIGNMENT assignment;
  assignment.fft_length = 0;
  char *tail;
  LINE_BUFFER line;
  unsigned int linecount=0;

  f_in = _fopen(filename, "r");
  if(NULL == f_in)
  {
//    printf("Can't open workfile %s in %s\n", filename, getcwd(line,sizeof(LINE_BUFFER)) );
    fprintf(stderr, "Can't open workfile %s\n", filename);
    return CANT_OPEN_FILE;	// nothing to open...
  }
  do
  {
    linecount++;
    
    value = parse_worktodo_line(f_in,&assignment,&line,&tail);
    
    if ((BLANK_LINE == value) || (NONBLANK_LINE == value))
      continue;
    if (NO_WARNING == value)
    {
      if (valid_assignment(assignment.exponent, assignment.fft_length))
        break;
      value = INVALID_DATA;
    }

    if (END_OF_FILE == value)
      break;
    fprintf(stderr, "Warning: ignoring line %u: \"%s\" in \"%s\". Reason: ", linecount, line, filename);
    switch(value)
    {
      case LONG_LINE:           printf("line is too long.\n"); break;
      case NO_TEST_EQUAL:       printf("doesn't begin with Test= or DoubleCheck=.\n");break;
      case INVALID_FORMAT:      printf("invalid format.\n");break;
      case INVALID_DATA:        printf("invalid data.\n");break;
      default:                  printf("unknown error.\n"); break;
    }
    
    // if (LONG_LINE != value)
    //	return 2;
  }
  while (1);
  
  fclose(f_in);
  if (NO_WARNING == value)
  {
    *exponent = assignment.exponent;
    if(assignment.fft_length > 0) *fft_length = assignment.fft_length;
    
    if (key!=NULL)strcopy(*key, assignment.hex_key, MAX_LINE_LENGTH+1);
    
    return OK;
  }
  else
    fprintf(stderr, "No valid assignment found.\n\n");
    return VALID_ASSIGNMENT_NOT_FOUND;
}

/************************************************************************************************************
 * Function name : clear_assignment                                                                         *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		int exponent									            *
 *     OUTPUT :                                        							    *
 *                                                                                                          *
 *     0 - OK												    *
 *     3 - clear_assignment    : cannot open file <filename>						    *
 *     4 - clear_assignment    : cannot open file "__worktodo__.tmp"					    *
 *     5 - clear_assignment    : assignment not found							    *
 *     6 - clear_assignment    : cannot rename temporary workfile to regular workfile			    *
 *                                                                                                          *
 ************************************************************************************************************/
enum ASSIGNMENT_ERRORS clear_assignment(char *filename, int exponent)
{
  int found = 0;
  FILE *f_in, *f_out;
  LINE_BUFFER line;	// line buffer
  char *tail = NULL;	// points to tail material in line, if non-null
  enum PARSE_WARNINGS value;
  unsigned int line_to_drop = UINT_MAX;
  unsigned int current_line;
  struct ASSIGNMENT assignment;	// the found assignment....
  
  f_in = _fopen(filename, "r");
  if (NULL == f_in)
    return CANT_OPEN_WORKFILE;
  
  f_out = _fopen("__worktodo__.tmp", "w");
  if (NULL == f_out)
  {
    fclose(f_in);
    return CANT_OPEN_TEMPFILE;
  }

  current_line =0;
  while (END_OF_FILE != (value = parse_worktodo_line(f_in,&assignment,&line,&tail)) )
  {
    current_line++;
    if (NO_WARNING == value)
    {
      if( (exponent == assignment.exponent) )	// make final decision
      {
        if (line_to_drop > current_line)
        line_to_drop = current_line;
        break;
      }
      else
      {
        line_to_drop = current_line+1;	// found different assignment, can drop no earlier than next line
      }
    }
    else if ((BLANK_LINE == value) && (UINT_MAX == line_to_drop))
      line_to_drop = current_line+1;
  }

  
  errno = 0;
  if (fseek(f_in,0L,SEEK_SET))
  {
    fclose(f_in);
    f_in = _fopen(filename, "r");
    if (NULL == f_in)
    {
      fclose(f_out);
      return CANT_OPEN_WORKFILE;
    }
  }
  
  found = 0;
  current_line = 0;
  while (END_OF_FILE != (value = parse_worktodo_line(f_in,&assignment,&line,&tail)) )
  {
    current_line++;
    if ((NO_WARNING != value) || found)
    {
      if ((found) || (current_line < line_to_drop))
        fprintf(f_out, "%s\n", line);
    }
    else	// assignment on the line, so we may need to print it..
    {
      found = (exponent == assignment.exponent);
      if (!found)
      {
        fprintf(f_out,"%s\n",line);
      }
      else	// we have the assignment...
      {
        // Do nothing; we don't print this to the temp file, 'cause we're trying to delete it :)
      }
    }
  }	// while.....
  fclose(f_in);
  fclose(f_out);
  if (!found)
    return ASSIGNMENT_NOT_FOUND;
  if(remove(filename) != 0)
    return CANT_RENAME;
  if(rename("__worktodo__.tmp", filename) != 0)
    return CANT_RENAME;
  return OK;
}

/*!----------------------------------------------------------------------------------------------*/
/*! These functions are slightly more modified from Oliver's equivalents. */

int IniGetInt(char *inifile, char *name, int *value, int dflt)
{
  FILE *in;
  char buf[100];
  int found=0;
  in=_fopen(inifile,"r");
  if(!in) goto error;
  while(fgets(buf,100,in) && !found)
  {
    if(!strncmp(buf,name,strlen(name)) && buf[strlen(name)]=='=')
    {
      if(sscanf(&(buf[strlen(name)+1]),"%d",value)==1)found=1;
    }
  }
  fclose(in);
  if(found) return 1;
  error: *value = dflt; return 0;
}

int IniGetStr(char *inifile, char *name, char *string, char* dflt)
{
  FILE *in;
  char buf[100];
  int found=0;
  in=_fopen(inifile,"r");
  if(!in)
    goto error;
  while(fgets(buf,100,in) && !found)
  {
    if(!strncmp(buf,name,strlen(name)) && buf[strlen(name)]=='=')
    {
      if(sscanf(&(buf[strlen(name)+1]),"%131s",string)==1)found=1; // CuLu's char*'s are 132 bytes
    }
  }
  fclose(in);
  if(found)return 1;
  error: _sprintf(string, "%s", dflt); return 0;
}
