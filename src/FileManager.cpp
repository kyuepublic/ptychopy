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

#include "FileManager.h"
#include "DiffractionLoader.h"
#include "ThreadPool.h"
#include "Parameters.h"
#include <algorithm>
#include  <fcntl.h>
#include <sys/mman.h>

using namespace std;

#ifdef HAVE_HDF5
	#ifdef USE_SINGLE_PRECISION
	#define H5T_real_t H5T_NATIVE_FLOAT
	#else
	#define H5T_real_t H5T_NATIVE_DOUBLE
	#endif
#include <hdf5.h>

/* File Formats */

//HDF5 file

void H5File::openFile(const string& fname)
{
	if(!m_open)
	{
		m_fileID = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		if(m_fileID < 0)
		{
			fprintf(stderr, "ERROR: Failed to open file (%s)\n", fname.c_str());
			return;
		}
		const PreprocessingParams* pParams = CXParams::getInstance()->getPreprocessingParams();
		m_datasetID = H5Dopen2(m_fileID, pParams->hdf5DatasetName.c_str(), H5P_DEFAULT);
		if(m_datasetID < 0)
		{
			fprintf(stderr, "ERROR:  Failed to open HDF5 dataset (%s)\n", pParams->hdf5DatasetName.c_str());
			return;
		}
		m_dataspaceID = H5Dget_space(m_datasetID);
		if(m_dataspaceID < 0)
		{
			fprintf(stderr, "ERROR: Could not retrieve HDF5 dataspace\n");
			return;
		}
		m_datasetRank = H5Sget_simple_extent_ndims(m_dataspaceID);
		if(m_datasetRank < 2 || m_datasetRank > 3)
		{
			fprintf(stderr, "ERROR: HDF5 dataset must either contain 2D or 3D arrays.\n");
			return;
		}
		if(m_datasetRank == 3)
		{
			hsize_t dims_out[3];
			H5Sget_simple_extent_dims(m_dataspaceID, dims_out, NULL);
			hsize_t sdims[2] = {dims_out[1], dims_out[2]};
			m_sectionDsID = H5Screate_simple (2, sdims, NULL);
		}
		m_open = true;
	}
}

void H5File::closeFile()
{
	if(!m_open)
		return;
	if(m_datasetRank == 3)
		H5Sclose(m_sectionDsID);
	H5Sclose(m_dataspaceID);
	H5Dclose(m_datasetID);
	H5Fclose(m_fileID);
	for(size_t i=0; i<m_buffers.size();++i)
		delete[] m_buffers[i];
	m_buffers.clear();
	m_open = false;
}

real_t* H5File::readSection(unsigned int i, unsigned int& x, unsigned int& y)
{
	if(!m_open)
		return 0;

	hsize_t dims_out[2];
	H5Sget_simple_extent_dims((m_datasetRank==3)?m_sectionDsID:m_dataspaceID, dims_out, NULL);
	x = dims_out[0]; y = dims_out[1];
	real_t* h_buffer = new real_t[x*y];

	if(m_datasetRank == 2)
		H5Dread(m_datasetID, H5T_real_t, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_buffer);
	else if(m_datasetRank == 3)
	{
		hsize_t count[3] = {1,x,y};
		hsize_t offset[3]= {i,0,0};
		H5Sselect_hyperslab(m_dataspaceID, H5S_SELECT_SET, offset, NULL, count, NULL);
		H5Dread(m_datasetID, H5T_real_t, m_sectionDsID, m_dataspaceID, H5P_DEFAULT, h_buffer);
	}

	m_buffers.push_back(h_buffer);
	return h_buffer;
}
#endif

//Binary file
void BinFile::openFile(const string& fname)
{
	if(!m_open)
	{
		m_fileID = open(fname.c_str(), O_RDONLY);
		if(m_fileID == -1)
		{
			fprintf(stderr, "ERROR: Failed to open file (%s)\n", fname.c_str());
			return;
		}
		m_fileSize = lseek(m_fileID, 0, SEEK_END);
		m_mappedPtr = (real_t*)mmap(NULL, m_fileSize, PROT_READ, MAP_SHARED, m_fileID, 0);
		if(m_mappedPtr == MAP_FAILED)
		{
			fprintf(stderr, "ERROR: Failed to map file (%s)\n", fname.c_str());
			return;
		}
		m_open = true;
	}
}

void BinFile::closeFile()
{
	if(!m_open)
		return;
	close(m_fileID);
	munmap(m_mappedPtr, m_fileSize);
	m_open = false;
}

real_t* BinFile::readSection(unsigned int i, unsigned int& x, unsigned int& y)
{return m_open?m_mappedPtr+(i*x*y):0;}

//CSV file
void CSVFile::openFile(const string& fname)
{
	if(!m_open)
	{
		ifstream file;
		file.open(fname.c_str());
		if(!file.is_open())
		{
			fprintf(stderr, "ERROR: Failed to open file (%s)\n", fname.c_str());
			return;
		}
		m_fileContents.clear();
		string dataLine;
		real_t value; char comma;

		while(getline(file, dataLine))
		{
			istringstream iss(dataLine);
			while(!iss.eof())
			{
				iss >> value;
				iss >> comma;
				m_fileContents.push_back(value);
			}
		}
		file.close();
		m_open = true;
	}
}

void CSVFile::closeFile()
{
	m_fileContents.clear();
	m_open = false;
}

real_t* CSVFile::readSection(unsigned int i, unsigned int& x, unsigned int& y)
{return m_open?&(m_fileContents[0])+(i*x*y):0;}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
FileManager::FileManager()
{pthread_mutex_init ( &m_tasksMutex, NULL);}

FileManager::~FileManager()
{pthread_mutex_destroy(&m_tasksMutex);}

bool FileManager::addIOTask(const string& fname, unsigned int i, Cuda3DElement<real_t> db)
{
	pthread_mutex_lock(&m_tasksMutex);
	FileManager::taskIterator it = m_ioTasks.find(fname);
	IOFile* ioFile = 0;

	if(it == m_ioTasks.end()) //new file
	{
		int dotPos = fname.find_last_of('.');
		if(dotPos<=0)
		{
			fprintf(stderr, "Error in -fp formatting parameter!\nNo file extension found.\n");
			return false;
		}
		string extension = fname.substr(dotPos+1, fname.length()-dotPos);
		transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

		if(extension == "h5")
		#ifdef HAVE_HDF5
			ioFile = new H5File();
		#else
			fprintf(stderr, "No HDF5 support found!\nPlease use configure specify the location of the parallel HDF5 library in the build.sh script.\n");
		#endif
		else if(extension == "csv")
			ioFile = new CSVFile();
		else if(extension == "bin")
			ioFile = new BinFile();
		else
		{
			fprintf(stderr, "Unexpected file extension found in -fp parameter. Valid file formats are (*.h5, *.csv, *.bin)\n");
			return false;
		}
		m_ioTasks[fname] = ioFile;
	}
	else ioFile = m_ioTasks[fname];

	ioFile->addReader(IOTask(i, db));
	pthread_mutex_unlock(&m_tasksMutex);

	return true;
}

void FileManager::loadData()
{
	int currentGPUID;
	cudaGetDevice(&currentGPUID);
	const PreprocessingParams* pParams = CXParams::getInstance()->getPreprocessingParams();
	ThreadPool threadpool(pParams->io_threads);
	for(FileManager::taskIterator i=m_ioTasks.begin(); i!=m_ioTasks.end(); ++i)
		for(size_t j=0; j<i->second->getReadersNum(); ++j)
			threadpool.addTask(new DiffractionLoader(currentGPUID, i->first.c_str(),
													i->second->getReader(j).deviceBuffer,
													i->second->getReader(j).index, *pParams));
	threadpool.start();
	threadpool.finish();
}

real_t* FileManager::readBuffer(const string& fname, unsigned int i, unsigned int& x, unsigned int& y)
{
	pthread_mutex_lock(&m_tasksMutex);
	m_ioTasks[fname]->openFile(fname);
	pthread_mutex_unlock(&m_tasksMutex);

	if(!m_ioTasks[fname]->isOpen())
		return 0;

	return m_ioTasks[fname]->readSection(i, x, y);
}

void FileManager::bufferDone(const string& fname)
{
	pthread_mutex_lock(&m_tasksMutex);
	m_ioTasks[fname]->removeReader();
	if(m_ioTasks[fname]->getReadersNum()==0)
	{
		m_ioTasks[fname]->closeFile();
		m_ioTasks.erase(fname);
	}
	pthread_mutex_unlock(&m_tasksMutex);
}
