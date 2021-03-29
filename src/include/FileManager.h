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

#ifndef FILEMANAGER_H_
#define FILEMANAGER_H_

#include "Singleton.h"
#include "Cuda2DArray.hpp"
#include "datatypes.h"
#include <pthread.h>
#include <string>
#include <map>
#include <vector>


struct IOTask
{
	unsigned int index;
	Cuda3DElement<real_t> deviceBuffer;

	IOTask(unsigned int i=0, Cuda3DElement<real_t> db=Cuda3DElement<real_t>(0,0,0)) : index(i), deviceBuffer(db)
	{}
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class IOFile
{
protected:
	bool m_open;
	std::vector<IOTask> m_readers;

public:
	IOFile() : m_open(false){}
	virtual ~IOFile(){}

	virtual bool isOpen() 						const {return m_open;}
	virtual size_t getReadersNum() 				const {return m_readers.size();}
	virtual const IOTask& getReader(size_t i) 	const {return m_readers[i];}
	virtual void addReader(IOTask t)				  {m_readers.push_back(t);}
	virtual void removeReader()						  {m_readers.pop_back();}
	virtual void openFile(const std::string&) 								= 0;
	virtual void closeFile() 												= 0;
	virtual real_t* readSection(unsigned int,unsigned int&,unsigned int&)	= 0;

};
#define HAVE_HDF5

#ifdef HAVE_HDF5
class H5File : public IOFile
{
private:
	int m_fileID;
	int m_datasetID;
	int m_dataspaceID;
	int m_sectionDsID;
	int m_datasetRank;
	std::vector<real_t*> m_buffers;

public:
	H5File() : IOFile(), m_fileID(0), m_datasetID(0), m_dataspaceID(0), m_sectionDsID(0), m_datasetRank(0), m_buffers()
	{}
	~H5File(){}

	void openFile(const std::string&);
	void closeFile();
	real_t* readSection(unsigned int,unsigned int&,unsigned int&);
};
#endif

class BinFile : public IOFile
{
private:
	int m_fileID;
	size_t m_fileSize;
	real_t* m_mappedPtr;

public:
	BinFile() : IOFile(), m_fileID(0), m_fileSize(0), m_mappedPtr(0)
	{}
	~BinFile(){}

	void openFile(const std::string&);
	void closeFile();
	real_t* readSection(unsigned int,unsigned int&,unsigned int&);
};

class CSVFile : public IOFile
{
private:
	std::vector<real_t> m_fileContents;

public:
	CSVFile() : IOFile(), m_fileContents()
	{}
	~CSVFile(){}

	void openFile(const std::string&);
	void closeFile();
	real_t* readSection(unsigned int,unsigned int&,unsigned int&);
};
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FileManager
{
private:
	pthread_mutex_t m_tasksMutex;
	std::map<std::string, IOFile*> m_ioTasks;
	typedef std::map<std::string, IOFile*>::iterator taskIterator;

public:
	FileManager();
	virtual ~FileManager();

	bool addIOTask(const std::string& fname, unsigned int i,  Cuda3DElement<real_t> db);
	void loadData(); //Blocking function until all the data is read
	real_t* readBuffer(const std::string&, unsigned int,unsigned int&,unsigned int&);
	void bufferDone(const std::string& fname);
};

typedef Singleton<FileManager> IO;

#endif /* FILEMANAGER_H_ */
