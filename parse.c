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

#ifndef linux
#define strncasecmp _strnicmp
#define sscanf sscanf_s /* This last one only works for scanning numbers, */
#endif			/* or strings with a defined length (e.g. "%131s") */

#define MAX_LINE_LENGTH 100

void strcopy(char* dest, char* src, size_t n) 
{
#ifdef linux
	strncpy(dest, src, n);
#else
	strncpy_s(dest, MAX_LINE_LENGTH+1, src, n);
#endif
}

FILE* _fopen(const char* path, const char* mode) {
#ifdef linux
	return fopen(path, mode);
#else
	FILE* stream;
	errno_t err = fopen_s(&stream, path, mode);
	if(err) return NULL;
	else return stream;
#endif
}

int file_exists(char* name) {
	FILE* stream;
	if((stream = _fopen(name, "r")))
	  {  
	    fclose(stream);
	    return 1;
	  }
	else return 0;
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

int valid_assignment(int exp)
/*
returns 1 if the assignment is within the supported bounds of CUDALucas,
0 otherwise.
*/
{
  int ret = 1;
  
       if(exp < 86243)      {ret = 0; fprintf(stderr, "Warning: exponents < 86243 are not supported!\n");}
  else if(!isprime(exp))      {ret = 0; fprintf(stderr, "Warning: exponent is not prime!\n");}
  //! Perhaps add a largest exponent?  
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
	char assignment_key[MAX_LINE_LENGTH+1];	// optional assignment key....
	char comment[MAX_LINE_LENGTH+1];	// optional comment.
						// if it has a newline at the end, it was on a line by itself preceding the assignment.
						// otherwise, it followed the assignment on the same line.
};
//! Should we include FFT length in this?

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
  printf("Entered parse_worktodo.\n");
  #endif
  char line[MAX_LINE_LENGTH+1], *ptr, *ptr_start, *ptr_end;
  int c;	// extended char pulled from stream;

  unsigned int scanpos;
  unsigned int number_of_commas;

  enum PARSE_WARNINGS reason = NO_WARNING;

  int proposed_exponent;
  #ifdef EBUG
  printf("About to call fgets().\n");
  #endif
  if(NULL==fgets(line, MAX_LINE_LENGTH+1, f_in))
  {
    return END_OF_FILE;
  }
  #ifdef EBUG
  printf("Called fgets().\n");
  #endif
  if (linecopy != NULL)	{ // maybe it wasn't needed....
    strcopy(*linecopy, line, MAX_LINE_LENGTH+1);	// this is what was read...
    if( NULL != strchr(*linecopy, '\n') )
      *strchr(*linecopy, '\n') = '\0';
  }
  if((strlen(line) == MAX_LINE_LENGTH) && (!feof(f_in)) && (line[strlen(line)-1] !='\n') ) // long lines disallowed,
  {
    reason = LONG_LINE;
    do
    {
      c = fgetc(f_in);
      if ((EOF == c) ||(iscntrl(c)))	// found end of line
        break;
    }
    while(1);
  }

  if (linecopy != NULL)
    *endptr = *linecopy;	// by default, non-significant content is whole line
  #ifdef EBUG
  printf("Starting to parse line...\n");
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
  #ifdef EBUG
  printf("Found a Test=...\n");
  #endif
  ptr = 1+ strstr(ptr,"=");	// don't rescan..
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// ignore blanks...
    ptr++;
  number_of_commas = 0;
  for(scanpos = 0; scanpos < strlen(ptr); scanpos++)
  {
    if(ptr[scanpos] == ',')
      number_of_commas++; // count the number of ',' in the line
    if ((ptr[scanpos] == '\\') && (ptr[scanpos+1] == '\\'))
      break;	// comment delimiter
    if ((ptr[scanpos] == '/') && (ptr[scanpos+1] == '/'))
      break;	// //comment delimiter
    if (ptr[scanpos] == '#')
      break;    // comment delimiter
  }
  #ifdef EBUG
  printf("Counted commas...\n");
  #endif
  if ((number_of_commas > 3))	// must have less than 4 commas...
                            //! If we implement FFT in line, we will need 3 commas
    return INVALID_FORMAT;
    
  if(number_of_commas <= 1) {
    assignment->assignment_key[0] = '\0';
  } else
  {  
    strcopy(assignment->assignment_key,ptr,1+(strstr(ptr,",")-ptr) );	// copy the comma..
    *strstr(assignment->assignment_key,",") = '\0';	// null-terminate key
    ptr=1 + strstr(ptr,",");
  }
  #ifdef EBUG
  printf("Copied AID, parsing expo...\n");
  #endif
  // ptr now points at exponent...
  ptr_start = ptr;
  while( (isspace(*ptr_start)) && ('\0' != *ptr_start ))
    ptr_start++;
  if ('M' == *ptr_start)	// M means Mersenne exponent...
    ptr_start++;
  errno = 0;
  proposed_exponent = strtoul(ptr_start, &ptr_end, 10);
  if (ptr_start == ptr_end)
    return INVALID_FORMAT;	// no conversion
  if ((0!=errno) || (proposed_exponent > UINT_MAX))
    return INVALID_DATA;	// for example, too many digits.
  ptr=ptr_end;
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// ignore blanks...
    ptr++;
  if (NULL != strstr(ptr,"\n"))		// kill off any trailing newlines...
    *strstr(ptr,"\n") = '\0';
  if (*ptr != '\0')
    strcopy(assignment->comment, ptr, 1+strchr(ptr,'\0')-ptr);
  #ifdef EBUG
  printf("Parsed exponent. Returning from parse_worktodo.\n");
  #endif

  if (linecopy != NULL)
    *endptr = *linecopy + (ptr_end - line);
  //! I'm not sure what this is intended to do...

  assignment->exponent = proposed_exponent;
  return reason;
}

/************************************************************************************************************
 * Function name : get_next_assignment                                                                      *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		int *exponent									    	    *
 *		char *assignment_key[100];								    *
 *     OUTPUT :                                        							    *
 *                                                                                                          *
 *     0 - OK												    *
 *     1 - get_next_assignment : cannot open file							    *
 *     2 - get_next_assignment : no valid assignment found						    *
 ************************************************************************************************************/
enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, int *exponent, LINE_BUFFER *key, int verbosity)
{
  #ifdef EBUG
  printf("Starting GNA.\n");
  #endif
  FILE *f_in;
  enum PARSE_WARNINGS value;
  struct ASSIGNMENT assignment;
  char *tail;
  LINE_BUFFER line;
  unsigned int linecount=0;

  #ifdef EBUG
  printf("Opening %s...\n", filename);
  #endif
  f_in = _fopen(filename, "r");
  if(NULL == f_in)
  {
//    printf("Can't open workfile %s in %s\n", filename, getcwd(line,sizeof(LINE_BUFFER)) );
    fprintf(stderr, "Can't open workfile %s\n", filename);
    return CANT_OPEN_FILE;	// nothing to open...
  }
  #ifdef EBUG
  printf("Opened %s. About to call parse_worktodo.\n", filename);
  #endif
  do
  {
    linecount++;
    value = parse_worktodo_line(f_in,&assignment,&line,&tail);
    #ifdef EBUG
    printf("Got an assignment. Retval: %d. Expo: %d\n", value, assignment.exponent);
    #endif
    if ((BLANK_LINE == value) || (NONBLANK_LINE == value))
      continue;
    if (NO_WARNING == value)
    {
      if (valid_assignment(assignment.exponent))
        break;
      value = INVALID_DATA;
    }

    if (END_OF_FILE == value)
      break;
    if(verbosity >= 1)
    {
      fprintf(stderr, "Warning: ignoring line %u: \"%s\" in \"%s\". Reason: ", linecount, line, filename);
      switch(value)
      {
        case LONG_LINE:           printf("line is too long.\n"); break;
        case NO_TEST_EQUAL:       printf("doesn't begin with Test= or DoubleCheck=.\n");break;
        case INVALID_FORMAT:      printf("invalid format.\n");break;
        case INVALID_DATA:        printf("invalid data.\n");break;
        default:                  printf("unknown error.\n"); break;
      }
    }
    
    // if (LONG_LINE != value)
    //	return 2;
  }
  while (1);
  
  fclose(f_in);
  if (NO_WARNING == value)
  {
    *exponent = assignment.exponent;
    
    if (key!=NULL)strcopy(*key, assignment.assignment_key, MAX_LINE_LENGTH+1);
    
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
      if(sscanf(&(buf[strlen(name)+1]),"%131s",string)==1)found=1; //! CuLu's char*'s are 132 bytes
    }
  }
  fclose(in);
  if(found)return 1;
  error: string = dflt; return 0;
}
