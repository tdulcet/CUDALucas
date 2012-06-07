/*
This file used to be part of mfaktc.
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)
This file has been written by Luigi Morelli (L.Morelli@mclink.it)

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

#define MAX_LINE_LENGTH 100
typedef char LINE_BUFFER[MAX_LINE_LENGTH+1];

/*! We must declare these as as such because CUDA is recognized as C++, a different language. See:
 http://forums.nvidia.com/index.php?showtopic=190973 */
extern "C" enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, int *exponent, LINE_BUFFER *assignment_key, int verbosity);
extern "C" enum ASSIGNMENT_ERRORS clear_assignment(char *filename, int exponent);
extern "C" int valid_assignment(int exp);	// nonzero if assignment is valid
extern "C" int IniGetInt(char* ini_file, char* name, int* value, int dfault);
extern "C" int IniGetStr(char* ini_file, char* name, char* str, char* dfault);
extern "C" int file_exists(char* filename); // nonzero if file exists
