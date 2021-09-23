////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Copyright © 2019, UChicago Argonne, LLC
//
//All Rights Reserved
//
//Software Name: ptychopy
//
//By: Argonne National Laboratory
//
//OPEN SOURCE LICENSE
//
//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
//following conditions are met:
//
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
//disclaimer.
//
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
//disclaimer in the documentation and/or other materials provided with the distribution.
//
//3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
//derived from this software without specific prior written permission.
//
//DISCLAIMER
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
//WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef TIMER_H_
#define TIMER_H_

#ifdef WIN32   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif

class Timer
{
private:
    double m_startTimeInMicroSec;                 // starting time in micro-second
    double m_endTimeInMicroSec;                   // ending time in micro-second
    bool    m_stopped;                            // stop flag
#ifdef WIN32
    LARGE_INTEGER m_frequency;                    // ticks per second
    LARGE_INTEGER m_startCount;                   //
    LARGE_INTEGER m_endCount;                     //
#else
    timeval m_startCount;                         //
    timeval m_endCount;                           //
#endif

public:
    Timer();                                    // default constructor
    ~Timer(){}                                  // default destructor

    void   start();                             // start timer
    void   stop();                              // stop the timer
    double getElapsedTime();                    // get elapsed time in second
    double getElapsedTimeInSec();               // get elapsed time in second (same as getElapsedTime)
    double getElapsedTimeInMilliSec();          // get elapsed time in milli-second
    double getElapsedTimeInMicroSec();          // get elapsed time in micro-second
};

#endif // TIMER_H_
