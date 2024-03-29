#!/bin/sh
#####################################################-*-mode:shell-script-*-
##                                                                       ##
##                     Carnegie Mellon University                        ##
##                        Copyright (c) 2007                             ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  CARNEGIE MELLON UNIVERSITY AND THE CONTRIBUTORS TO THIS WORK         ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL CARNEGIE MELLON UNIVERSITY NOR THE CONTRIBUTORS BE LIABLE      ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
## adds up the duration (in minutes) for a given prompt file             ##
##                                                                       ##
###########################################################################

LANG=C; export LANG

if [ $# = 0 ]
then
   echo "Display size in minutes of databases"
   echo "Usage:  bin/find_db_duration scp_file wav_dir"
   exit 1
fi

if [ ! "$ESTDIR" ]
then
   echo "environment variable ESTDIR is unset"
   echo "set it to your local speech tools directory e.g."
   echo '   bash$ export ESTDIR=/home/awb/projects/speech_tools/'
   echo or
   echo '   csh% setenv ESTDIR /home/awb/projects/speech_tools/'
   exit 1
fi

awk '{print $1}' $1 |
while read x
do
   $ESTDIR/bin/ch_wave -info $2/$x.wav
done |
grep uration |
awk '{t+=$NF} 
     END {seconds = t;
          minutes = t/60.0;
          hours = t/(60.0*60.0);
          nhours = (t-(t%3600))/3600;
          nmins = ((t-(nhours*3600))-(t%60))/60;
          nsecs = t-(nhours*3600)-(nmins*60);
          printf("%02d:%02d:%02d or %0.2f hours or %0.2f minutes or %d seconds\n",nhours,nmins,nsecs,hours,minutes,seconds);}'
