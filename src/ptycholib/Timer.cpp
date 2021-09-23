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
#include "Timer.h"
#include <stdlib.h>

Timer::Timer(): m_startTimeInMicroSec(0), m_endTimeInMicroSec(0), m_stopped(true)
{
	#ifdef WIN32
	QueryPerformancem_frequency(&m_frequency);
	m_startCount.QuadPart = 0;
	m_endCount.QuadPart = 0;
	#else
	m_startCount.tv_sec = m_startCount.tv_usec = 0;
	m_endCount.tv_sec = m_endCount.tv_usec = 0;
	#endif
}

void Timer::start()
{
	if(m_stopped)
	{
		m_stopped = false; // reset stop flag
		#ifdef WIN32
		QueryPerformanceCounter(&m_startCount);
		#else
		gettimeofday(&m_startCount, NULL);
		#endif
	}
}

void Timer::stop()
{
	if(!m_stopped)
	{
		m_stopped = true; // set timer stopped flag
		#ifdef WIN32
		QueryPerformanceCounter(&m_endCount);
		#else
		gettimeofday(&m_endCount, NULL);
		#endif
	}
}

double Timer::getElapsedTimeInMicroSec()
{
	#ifdef WIN32
	if(!m_stopped)
		QueryPerformanceCounter(&m_endCount);

	m_startTimeInMicroSec = m_startCount.QuadPart * (1000000.0 / m_frequency.QuadPart);
	m_endTimeInMicroSec = m_endCount.QuadPart * (1000000.0 / m_frequency.QuadPart);
	#else
	if (!m_stopped)
		gettimeofday(&m_endCount, NULL);

	m_startTimeInMicroSec = (m_startCount.tv_sec * 1000000.0) + m_startCount.tv_usec;
	m_endTimeInMicroSec = (m_endCount.tv_sec * 1000000.0) + m_endCount.tv_usec;
	#endif

	return m_endTimeInMicroSec - m_startTimeInMicroSec;
}

double Timer::getElapsedTimeInMilliSec()
{return this->getElapsedTimeInMicroSec() * 0.001;}

double Timer::getElapsedTimeInSec()
{return this->getElapsedTimeInMicroSec() * 0.000001;}

double Timer::getElapsedTime()
{return this->getElapsedTimeInSec();}
