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
#include "ThreadPool.h"
#include "IRunnable.h"
#include <unistd.h>

using namespace std;

ThreadPool::ThreadPool(unsigned int size) : m_poolSize(size), m_tasksLeft(0)
{
	pthread_mutex_init ( &m_tasksMutex, NULL);
	pthread_mutex_init ( &m_tasksCounterMutex, NULL);
	pthread_cond_init (&m_allDoneCondition, NULL);
}

ThreadPool::~ThreadPool()
{
	pthread_mutex_destroy(&m_tasksMutex);
	pthread_mutex_destroy(&m_tasksCounterMutex);
	pthread_cond_destroy(&m_allDoneCondition);
}

void ThreadPool::start()
{
	if(m_tasksLeft==0)
		return;
	for(unsigned int i=0; i<m_poolSize; ++i)
	{
		pthread_t workerThread;
		pthread_create(&workerThread, NULL, &ThreadPool::threadFunc, (void *) this );
	}
}

void ThreadPool::addTask(IRunnable* task)
{
	pthread_mutex_lock(&m_tasksCounterMutex);
	m_tasksLeft++;
	pthread_mutex_unlock(&m_tasksCounterMutex);

	pthread_mutex_lock(&m_tasksMutex);
	m_tasks.push(task);
	pthread_mutex_unlock(&m_tasksMutex);
}

bool ThreadPool::getTask(IRunnable** currentTask)
{
	bool gotTask = false;

	pthread_mutex_lock(&m_tasksMutex);
	if(m_tasks.size()>0)
	{
		*currentTask = m_tasks.front();
		m_tasks.pop();
		gotTask = true;
	}
	pthread_mutex_unlock(&m_tasksMutex);

	return gotTask;
}

void ThreadPool::finish()
{
	pthread_mutex_lock(&m_tasksCounterMutex);
	while(m_tasksLeft>0)
		pthread_cond_wait(&m_allDoneCondition, &m_tasksCounterMutex);
	pthread_mutex_unlock(&m_tasksCounterMutex);
}

void ThreadPool::taskDone()
{
	pthread_mutex_lock(&m_tasksCounterMutex);
	m_tasksLeft--;
	if(m_tasksLeft==0)
		pthread_cond_signal(&m_allDoneCondition);
	pthread_mutex_unlock(&m_tasksCounterMutex);
}

void* ThreadPool::threadFunc(void *param)
{
	ThreadPool* myPool = (ThreadPool*)param;
	IRunnable* task = 0;
	while(myPool->getTask(&task))
	{
		if(task)
		{
			task->run();
			myPool->taskDone();
			delete task;
			task = 0;
		}
		//usleep(1);
	}
	return 0;
}
