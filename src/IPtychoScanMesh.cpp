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

#include "IPtychoScanMesh.h"
#include "Cuda2DArray.hpp"
#include "ScanMesh.cuh"
#include "Parameters.h"
#include "utilities.h"

#include <vector>
using namespace std;

IPtychoScanMesh::IPtychoScanMesh(CudaSmartPtr devPtr, real_t sX, real_t sY, unsigned int r): 	m_scanPositions(devPtr),
																								m_stepSizeX(sX),
																								m_stepSizeY(sY),
																								m_jitterRadius(r)
{
	if(m_jitterRadius>0)
	{
		//Allocation of the CUDA random states
		m_randStates = new Cuda2DArray<curandState>(m_scanPositions->getX(), m_scanPositions->getY());
		h_initRandomStates(m_randStates->getX(), m_randStates->getAlignedY(), m_randStates->getDevicePtr<curandState>());
	}
	m_meshDimensions.x = 0;
	m_meshDimensions.y = 0;
	m_meshOffsets.x = 0;
	m_meshOffsets.y = 0;
	m_maxima.x = 0;
	m_maxima.y = 0;
	m_minima.x = 0;
	m_minima.y = 0;

	m_Np_o_new.x=0;
	m_Np_o_new.y=0;

	m_miniter=0;
	m_maxNpos=0;
}

void IPtychoScanMesh::clear()
{
	m_indeces.clear();
	m_positions.clear();
	m_positions_o.clear();
}

unsigned int IPtychoScanMesh::getTotalScanPositionsNum() const
{return m_scanPositions->getNum();}

void IPtychoScanMesh::calculateMeshDimensions(unsigned int probeSize)
{
//	float extra = 0.2;
//	float2 Np_o= make_float2(1e5f, 1e5f);
//	int Npos=m_scanPositions->getX()*m_scanPositions->getY();
//	float Np_p = probeSize;



	m_maxima = h_maxFloat2(m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(),
							m_scanPositions->getY(), m_scanPositions->getAlignedY());
	m_minima = h_minFloat2(m_scanPositions->getDevicePtr<float2>(), m_scanPositions->getX(),
							m_scanPositions->getY(), m_scanPositions->getAlignedY());

	m_meshDimensions.x = ceil( (m_maxima.x-m_minima.x) + probeSize);
	m_meshDimensions.y = ceil( (m_maxima.y-m_minima.y) + probeSize);
}


void IPtychoScanMesh::addScanPosition(float2 scanPos, unsigned int index, bool flip)
{
	float2 tempPos = scanPos;
	scanPos.x=(flip?tempPos.y-m_minima.y : tempPos.x-m_minima.x);
	scanPos.y=(flip?tempPos.x-m_minima.x : tempPos.y-m_minima.y);
	m_positions.push_back(scanPos);
	m_indeces.push_back(index);
}

void IPtychoScanMesh::addScanPositionMLs(float2 scanPos, unsigned int index, bool flip)
{
	float2 tempPos = scanPos;
//	scanPos.x=(flip?tempPos.y-m_minima.y : tempPos.x-m_minima.x);
//	scanPos.y=(flip?tempPos.x-m_minima.x : tempPos.y-m_minima.y);
	scanPos.x=tempPos.x;
	scanPos.y=tempPos.y;
	m_positions.push_back(scanPos);
	m_positions_o.push_back(scanPos);
	m_indeces.push_back(index);
}

void IPtychoScanMesh::find_reconstruction_ROI(std::vector <uint2>& oROI1, std::vector <uint2>& oROI2,
		std::vector < std::vector <int> >& oROI_vec1, std::vector < std::vector <int> >& oROI_vec2)
{
	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	oROI1.clear();
	oROI2.clear();
	oROI_vec1.clear();
	oROI_vec2.clear();
	uint2 Np_p = make_uint2(rParams->desiredShape, rParams->desiredShape);
	uint2 Np_o = m_Np_o_new;
	uint2 tmp;
	uint2 tmp1;
	uint2 tmp2;
	float position1, position2;
	oROI_vec1.resize(m_positions.size());
	oROI_vec2.resize(m_positions.size());
	float2 ceilvar=make_float2(ceil(Np_o.x*1.0/2-Np_p.x*1.0/2), ceil(Np_o.y*1.0/2-Np_p.y*1.0/2));

	m_sub_px_shift.clear();
	m_sub_px_shift.resize(m_positions.size());
	m_nsub_px_shift.clear();
	m_nsub_px_shift.resize(m_positions.size());

	for(int i=0; i<m_positions.size(); i++)
	{
		position1=m_positions[i].x+ceilvar.x;
		position2=m_positions[i].y+ceilvar.y;

		tmp.x=round(position1);
		tmp.y=round(position2);
		m_sub_px_shift[i]=make_float2((position2-tmp.y), (position1-tmp.x));
		m_nsub_px_shift[i]=make_float2(-1*(position2-tmp.y), -1*(position1-tmp.x));

		tmp1.x=tmp.x-1;
		tmp1.y=tmp.x+Np_p.x-2;
		oROI1.push_back(tmp1);
		std::vector <int> tmp1vec(tmp1.y-tmp1.x+1);
		for(int j=0; j<tmp1vec.size(); j++)
			tmp1vec[j]=tmp1.x+j;
		oROI_vec1[i]=tmp1vec;
		tmp2.x=tmp.y-1;
		tmp2.y=tmp.y+Np_p.y-2;
		oROI2.push_back(tmp2);
		std::vector <int> tmp2vec(tmp2.y-tmp2.x+1);
		for(int j=0; j<tmp2vec.size(); j++)
			tmp2vec[j]=tmp2.x+j;
		oROI_vec2[i]=tmp2vec;
	}

}
void IPtychoScanMesh::find_reconstruction_ROI_O()
{
	find_reconstruction_ROI(m_oROI1, m_oROI2, m_oROI_vec1, m_oROI_vec2);

}

void IPtychoScanMesh::precalculate_ROI()
{
	uint2 Np_o = m_Np_o_new;
	m_object_ROIx=make_uint2(round(Np_o.x*0.5+m_minima.x), round(Np_o.x*0.5+m_maxima.x)-1);
	m_object_ROIy=make_uint2(round(Np_o.y*0.5+m_minima.y), round(Np_o.y*0.5+m_maxima.y)-1);
}

void IPtychoScanMesh::initoROIvec()
{
	find_reconstruction_ROI(m_oROI1, m_oROI2, m_oROI_vec1, m_oROI_vec2);

}
void IPtychoScanMesh::scanPositionHist(point pt, unsigned int diffcount, std::vector<int>& vecnbins, std::vector<int>& vecbins)
{
	vecnbins.clear();
	vecbins.clear();

	std::vector<int> vecgroup, vecfullgroup;
	std::vector<float> vecHistGroup;

	for ( int i=0; i<diffcount; i++ )
	{
		vecgroup.push_back(pt[i].group);
		vecfullgroup.push_back(pt[i].group);
	}
	// sort the group and get the unique
	std::sort(vecgroup.begin(), vecgroup.end());
	std::sort(vecfullgroup.begin(), vecfullgroup.end());

	// Delete duplicate element in the array
	std::vector<int>::iterator it;
	it = std::unique (vecgroup.begin(), vecgroup.end());
	vecgroup.resize( std::distance(vecgroup.begin(),it) );

	vecHistGroup.push_back(0);
	vecnbins.push_back(0);

	for(int i=1; i<vecgroup.size(); i++)
	{
		vecHistGroup.push_back((vecgroup[i-1]+vecgroup[i])*1.0/2);
		vecnbins.push_back(0);
//		vecbins.push_back(vecgroup[i]);
	}
	// Matlab nbins test

	for(int i=0; i<vecgroup.size(); i++)
	{
		vecbins.push_back(vecgroup[i]);
	}

	for(int i=0; i<diffcount; i++)
	{
		if(vecfullgroup[i]<=vecHistGroup[1])
		{
			vecnbins[0]++;
		}
		else if(vecfullgroup[i]>vecHistGroup[vecHistGroup.size()-1])
		{
			vecnbins[vecHistGroup.size()-1]++;
		}
		else
		{
			for(int j=1; j<vecHistGroup.size()-1; j++)
			{
				if(vecfullgroup[i]>vecHistGroup[j]&&vecfullgroup[i]<=vecHistGroup[j+1])
					vecnbins[j]++;
			}
		}
	}


}

void IPtychoScanMesh::setScanPositions(std::vector<float2>& positions)
{
	m_positions.clear();
	m_positions=positions;

}

void IPtychoScanMesh::scanPositionMatlabHist(point pt, unsigned int diffcount, std::vector<int>& vecnbins, std::vector<int>& vecbins)
{
	vecnbins.clear();
	vecbins.clear();

	std::vector<int> vecgroup, vecfullgroup;
	std::vector<float> vecHistGroup;

	for ( int i=0; i<diffcount; i++ )
	{
		vecgroup.push_back(pt[i].group);
		vecfullgroup.push_back(pt[i].group);
	}
	// sort the group and get the unique
	std::sort(vecgroup.begin(), vecgroup.end());
	std::sort(vecfullgroup.begin(), vecfullgroup.end());

	// Delete duplicate element in the array
	std::vector<int>::iterator it;
	it = std::unique (vecgroup.begin(), vecgroup.end());
	vecgroup.resize( std::distance(vecgroup.begin(),it) );

	vecHistGroup.push_back(0);
	vecnbins.push_back(0);

	for(int i=1; i<vecgroup.size(); i++)
	{
		vecHistGroup.push_back((vecgroup[i-1]+vecgroup[i])*1.0/2);
		vecnbins.push_back(0);
//		vecbins.push_back(vecgroup[i]);
	}
	// Matlab nbins test

	for(int i=0; i<vecgroup.size(); i++)
	{
		vecbins.push_back(vecgroup[i]);
	}

	for(int i=0; i<diffcount; i++)
	{
		if(vecfullgroup[i]<=vecHistGroup[1])
		{
			vecnbins[0]++;
		}
		else if(vecfullgroup[i]>vecHistGroup[vecHistGroup.size()-1])
		{
			vecnbins[vecHistGroup.size()-1]++;
		}
		else
		{
			for(int j=1; j<vecHistGroup.size()-1; j++)
			{
				if(vecfullgroup[i]>vecHistGroup[j]&&vecfullgroup[i]<=vecHistGroup[j+1])
					vecnbins[j]++;
			}
		}
	}


}

double IPtychoScanMesh::scanPositionSD(std::vector<int>& vecnbins)
{

	double sum = 0.0, mean, variance = 0.0, stdDeviation;
	int size=vecnbins.size();

	for(int i=0; i<size; i++)
	   sum += vecnbins[i];

	mean=sum/size;

	for(int i=0; i<size; i++)
	   variance += pow(vecnbins[i] - mean, 2);

	variance=variance/size;
	stdDeviation = sqrt(variance);

	return stdDeviation;
}

void IPtychoScanMesh::updateScanPositionsDiff()
{
	float2 meanPositions;
	float2 meanPositions_o;
	float2 meanDiff;
	real_t sum1x=0, sum1y=0, sum2x=0, sum2y=0;
	size_t len=m_positions.size();
	for(int i=0; i<len; i++)
	{
		sum1x += m_positions[i].x;
		sum1y += m_positions[i].y;
		sum2x += m_positions_o[i].x;
		sum2y += m_positions_o[i].y;
	}
	meanPositions.x=sum1x/len;
	meanPositions.y=sum1y/len;
	meanPositions_o.x=sum2x/len;
	meanPositions_o.y=sum2y/len;

	meanDiff.x=meanPositions.x-meanPositions_o.x;
	meanDiff.y=meanPositions.y-meanPositions_o.y;

	for(int i=0; i<len; i++)
	{
		m_positions[i].x=m_positions[i].x-meanDiff.x;
		m_positions[i].y=m_positions[i].y-meanDiff.y;
	}

	int temp=1;
}

bool IPtychoScanMesh::loadTestIGroup(const char* filename, std::vector<int>& vecFile)
{
	vecFile.clear();
	std::ifstream infile1(filename, std::ofstream::in|std::ofstream::binary );


//	std::vector<int> groupsTest;

	std::string line1 = "";
	int rowIdx=0;
	int val1=0;
	while (std::getline(infile1, line1))
	{
        std::stringstream ss1(line1);
        int colIdx = 0;
        while(ss1 >> val1)
        {

        	vecFile.push_back(val1-1);
            if(ss1.peek() == ',') ss1.ignore();
            colIdx++;
        }
        rowIdx++;
	}

	infile1.close();
}

void IPtychoScanMesh::CalcMHWScore(std::vector<double>& scoresx, std::vector<double>& scoresy, double &x, double &y)
{

	size_t size = scoresx.size();

	sort(scoresx.begin(), scoresx.end());
	sort(scoresy.begin(), scoresy.end());
	if (size % 2 == 0)
	{
		x=1.0*(scoresx[size / 2 - 1] + scoresx[size / 2]) / 2;
		y=1.0*(scoresy[size / 2 - 1] + scoresy[size / 2]) / 2;
//	  return make_float2(x,y);
	}
	else
	{
		x=scoresx[size / 2];
		y=scoresy[size / 2];
//	  return make_float2(x,y);
	}

}


void IPtychoScanMesh::get_close_indices()
{

	int diffcount = m_gridDimensions.x*m_gridDimensions.y;
	int grouping = diffcount/100;
	int Ngroups=ceil((m_gridDimensions.x*m_gridDimensions.y)/(grouping*1.0));

	int diffIndex=0;
	point p, pt=(point)malloc(sizeof(point_t) * diffcount);
	std::vector<int> vecnbins;
	std::vector<double> vecScore;
	std::vector< std::vector<int> > vecGroup(10, std::vector<int> (diffcount) );
	std::vector< std::vector<point_t> > vecC(10, std::vector<point_t> (Ngroups) );
	std::vector<int> vecbinsall;

	for(int i=0; i<10; i++)
	{

		vecnbins.clear();
		vecbinsall.clear();
		diffIndex=diffcount;
		for (p = pt + diffcount; p-- > pt;)
		{
			diffIndex--;
			p->x = m_positions[diffIndex].x;
			p->y = m_positions[diffIndex].y;
			p->group = 0;
		}

		point c = lloyd(pt, diffcount, Ngroups);

		for(int j=0; j<diffcount; j++)
		{
			vecGroup[i][j]=pt->group;
		}

		for(int j=0; j<Ngroups; j++)
		{
			vecC[i][j]=c[j];
		}

		scanPositionHist(pt, diffcount, vecnbins, vecbinsall);

		double score=scanPositionSD(vecnbins);
		vecScore.push_back(score);
	}

	int scoreIndex=std::min_element(vecScore.begin(), vecScore.end())-vecScore.begin();

	// Get vecD, vecC[scoreIndex] vecGroup[scoreIndex]
	std::vector< std::vector<double> > vecD (diffcount, std::vector<double> (Ngroups) );
	double tempDist=0;

	for (int i=0; i<diffcount; i++)
	{
		for(int j=0; j<Ngroups; j++)
		{
			tempDist=pow((pt[i].x-vecC[scoreIndex][j].x),2)+pow((pt[i].y-vecC[scoreIndex][j].y),2);
			vecD[i][j]=tempDist;
		}
	}

	// For testing load from files

	char* filename1="/data2/JunjingData/groups.csv";
	std::ifstream infile1(filename1, std::ofstream::in|std::ofstream::binary );
	std::vector<int> groupsTest;
	std::string line1 = "";
	int rowIdx=0;
	int val1=0;
	while (std::getline(infile1, line1))
	{
        // Create a stringstream of the current line
        std::stringstream ss1(line1);
        int colIdx = 0;
        while(ss1 >> val1)
        {
        	groupsTest.push_back(val1-1);
            if(ss1.peek() == ',') ss1.ignore();
            colIdx++;
        }
        rowIdx++;
	}
	infile1.close();

	char* filename2="/data2/JunjingData/D.csv";
	std::ifstream infile2(filename2, std::ofstream::in|std::ofstream::binary );
	std::vector< std::vector<double> > dTest (diffcount, std::vector<double> (Ngroups) );
	std::string line2 = "";
	rowIdx=0;
	double val2=0;
	while (std::getline(infile2, line2))
	{
        // Create a stringstream of the current line
        std::stringstream ss2(line2);
        int colIdx = 0;
        while(ss2 >> val2)
        {
        	dTest[rowIdx][colIdx]=val2;
            if(ss2.peek() == ',') ss2.ignore();
            colIdx++;
        }
        rowIdx++;
	}
	infile2.close();

	char* filename3="/data2/JunjingData/C.csv";
	std::ifstream infile3(filename3, std::ofstream::in|std::ofstream::binary );
	std::vector< std::vector<double> > cTest (Ngroups, std::vector<double> (2) );
	std::string line3 = "";
	rowIdx=0;
	double val3=0;
	while (std::getline(infile3, line3))
	{
        // Create a stringstream of the current line
        std::stringstream ss3(line3);
        int colIdx = 0;
        while(ss3 >> val3)
        {
        	cTest[rowIdx][colIdx]=val3;
            if(ss3.peek() == ',') ss3.ignore();
            colIdx++;
        }
        rowIdx++;
	}
	infile3.close();

	// End of reading all three files of the value C D group

	// Find the best kmeans
	vecnbins.clear();
	diffIndex=diffcount;
	for (p = pt + diffcount; p-- > pt;)
	{
		diffIndex--;
		p->x = m_positions[diffIndex].x;
		p->y = m_positions[diffIndex].y;
		p->group = groupsTest[diffIndex];

	}

	int iter=0;

	while(true)
	{
		iter=iter+1;
		std::vector<int> vecbins;
		scanPositionMatlabHist(pt, diffcount, vecnbins, vecbins);
		std::vector<int> vectempbins(vecnbins);
		std::sort(vectempbins.begin(), vectempbins.end());
		std::vector<int>::iterator it;
		it = std::unique (vectempbins.begin(), vectempbins.end());
		int Ngroups_sizes=it-vectempbins.begin();

		int righit=ceil((iter*1.0)/1e3);
        if ( ( Ngroups_sizes <= std::max(2,righit) && ((Ngroups*grouping)!= diffcount || iter > 1e3 ))||  Ngroups_sizes == 1 )
            break;

    	int min_groupIndex=std::min_element(vecnbins.begin(), vecnbins.end())-vecnbins.begin();

    	std::vector<int> large_groups;

    	for(int i=0; i<Ngroups; i++)
    	{
    		if((vecnbins[i])>grouping)
    		{
    			large_groups.push_back(i);
    		}
    	}
    	if(large_groups.empty())
    		break;

    	std::vector<double> large_groups_all;
    	std::vector<int> large_groups_all_index;
    	for(int i=0; i<diffcount; i++)
    	{
    		for(int j=0; j<large_groups.size(); j++)
    		{
    			if(((pt+i)->group)==large_groups[j])
    			{
    				large_groups_all.push_back(dTest[i][min_groupIndex]);
    				large_groups_all_index.push_back(i);
    			}

    		}
    	}

//    	vector<int> testind;
//    	cout<<endl;

    	if(large_groups_all.empty())
    	{
    		continue;
    	}
    	else
    	{
    		double minGroup=*std::min_element(large_groups_all.begin(), large_groups_all.end());
    		for(int index=0; index<diffcount; index++)
    		{
    			if(dTest[index][min_groupIndex]==minGroup)
    			{
    				p=pt;
    				int ind_large=index;
    				p=p+ind_large;
    				p->group=min_groupIndex;
//    				cout<<index<<","<<endl;
//    				testind.push_back(index);
    			}
    		}

    	}
	}

// End of the while loop for updating the group pt y pt x to matlab

	for(int i=0; i<Ngroups; i++)
	{
		std::vector <double> vecmgroupx, vecmgroupy;
		double x=0, y=0;
		for(int j=0; j<diffcount; j++)
		{
			if(((pt+j)->group)==i)
			{
				vecmgroupx.push_back((pt+j)->x);
				vecmgroupy.push_back((pt+j)->y);
			}
		}

		CalcMHWScore(vecmgroupx, vecmgroupy, x, y);
		cTest[i][0]=y;	// x
		cTest[i][1]=x;	   // y

	}

	for(int i=0; i<Ngroups; i++)
	{
		for(int j=0; j<diffcount; j++)
		{
			dTest[j][i]=pow((pt+j)->y-cTest[i][0], 2)+pow((pt+j)->x-cTest[i][1], 2);
		}
	}

	std::vector <int> vecoptimal_group;
//	int sum=0;
//	cout<<endl;
	for(int i=0; i<diffcount; i++)
	{
		std::vector <double> tempvec=dTest[i];
		int sindindex=std::min_element(tempvec.begin(), tempvec.end())-tempvec.begin();
		vecoptimal_group.push_back(sindindex);
//		cout<<sindindex<<endl;
//		sum+=sindindex;
	}
	int nonoptimal_ratio_0 = 1;
	for(int iter=0; iter<10; iter++)
	{
		int optgroupSum=0;
		std::vector<int> ind_switch;

		for(int i=0; i<diffcount; i++)
		{
			if((pt+i)->group!=vecoptimal_group[i])
			{
				optgroupSum++;
				ind_switch.push_back(i);
			}
		}

		double nonoptimal_ratio=optgroupSum*1.0/diffcount;

		if (nonoptimal_ratio >= nonoptimal_ratio_0)
			break;

		nonoptimal_ratio_0 = nonoptimal_ratio;

		for(int i=0; i<optgroupSum; i++)
		{

			std::vector <double> maxcenter_dist;
			std::vector <double> center_dist;
			std::vector <int> maxcenter_dist_group, optimal_group_vec;
			for(int j=0; j<diffcount; j++)
			{
				center_dist.push_back(dTest[j][(pt+j)->group]);
			}

			for(int m=0; m<ind_switch.size(); m++)
			{
				maxcenter_dist.push_back(center_dist[ind_switch[m]]);
				maxcenter_dist_group.push_back((pt+ind_switch[m])->group);
				optimal_group_vec.push_back(vecoptimal_group[ind_switch[m]]);

			}

			int ind_worsedis=std::max_element(maxcenter_dist.begin(), maxcenter_dist.end())-maxcenter_dist.begin();
			int ind_worse=ind_switch[ind_worsedis];
			int group_old=maxcenter_dist_group[ind_worsedis];
			int group_new=optimal_group_vec[ind_worsedis];

			std::vector <double> minGroupNewOld;
			std::vector <int> minGroupNewOld_ind;
			for(int qindex=0; qindex<diffcount; qindex++)
			{
				if(((pt+qindex)->group)==group_new)
				{
					minGroupNewOld.push_back(dTest[qindex][group_old]);
					minGroupNewOld_ind.push_back(qindex);
				}
			}
			double minGNewOld=*std::min_element(minGroupNewOld.begin(), minGroupNewOld.end());

			int ind_new=-1;
			for(int qindex=0; qindex<diffcount; qindex++)
			{
				if(dTest[qindex][group_old]==minGNewOld)
				{
					ind_new=qindex;
					break;
				}

			}
//			int minGNewOldDis=std::min_element(minGroupNewOld.begin(), minGroupNewOld.end())-minGroupNewOld.begin();
//			int ind_new=minGroupNewOld_ind[minGNewOldDis];
// matlab code has different result of ind_worse ind_new due the double precision issue after this the rsult is different
			(pt+ind_worse)->group=group_new;
			(pt+ind_new)->group=group_old;

			ind_switch[ind_worsedis]=0;

			int zeroflag=0;
			for(int oindex=0; oindex<ind_switch.size(); oindex++)
			{
				if(ind_switch[oindex]==ind_new)
					ind_switch[oindex]=0;
				if(ind_switch[oindex]==0)
					zeroflag++;
			}

			if(zeroflag==ind_switch.size())
				break;


		}
	}

	std::vector<int> vecnbins1;
	std::vector<int> vecbins1;
	scanPositionMatlabHist(pt, diffcount, vecnbins1, vecbins1);

	vector<pair<int,int> > veca;

	for (int m = 0 ;m < vecnbins1.size() ; m++) {
		veca.push_back (make_pair (-vecnbins1[m],m)); //
	}

	std::sort (veca.begin(),veca.end());

	std::vector< std::vector<int> > indices_out (Ngroups, std::vector<int>() );
	std::vector< std::vector<int> > scan_ids_out (Ngroups, std::vector<int>() );
	std::vector<pair<int,int> > Nitems;

	for(int nindex=0; nindex<Ngroups; nindex++)
	{
//		cout<<nindex<<endl;
		std::vector <int> indices, scan_ids;
		int groupNo=vecbins1[veca[nindex].second];
		int lengthNitem=0;
		for(int m=0; m<diffcount; m++)
		{
			if(((pt+m)->group)==groupNo)
			{
//				indices.push_back(m);
//				scan_ids.push_back(1);
				indices_out[nindex].push_back(m);
				scan_ids_out[nindex].push_back(1);
				lengthNitem++;
//				cout<<m<<" ";
			}
		}

//		indices_out.push_back(indices);
//		scan_ids_out.push_back(scan_ids);
		Nitems.push_back(make_pair (-lengthNitem,nindex));
	}

	std::sort(Nitems.begin(), Nitems.end());
	if(!m_compactindices_out.empty())
		m_compactindices_out.clear();
	if(!m_compactscan_ids_out.empty())
		m_compactscan_ids_out.clear();

	m_compactindices_out.resize(Nitems.size());
	m_compactscan_ids_out.resize(Nitems.size());

	for(int nindex=0; nindex<Nitems.size(); nindex++)
	{
		int newindex=Nitems[nindex].second;
		m_compactindices_out[nindex]=indices_out[newindex];
		m_compactscan_ids_out[nindex]=scan_ids_out[newindex];
	}

}

void IPtychoScanMesh::get_nonoverlapping_indices()
{

	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	int diffcount = m_gridDimensions.x*m_gridDimensions.y;
	int grouping = diffcount/100;
//	int Ngroups=ceil(diffcount/(grouping*1.0));
	grouping = ceil(grouping*1.0/1);//par.Nscans could be 1 and any number
	int miniter= std::min(rParams->Niter, 30);
	m_miniter=miniter;
	int ind_start=0;
	int Npos_tmp=diffcount;
	int max_groups=0;
	int maxNpos=ceil(Npos_tmp*1.0/grouping);
	m_maxNpos=maxNpos;
	max_groups = std::max(max_groups, maxNpos);

	if(!m_sparseindices_out.empty())
		m_sparseindices_out.clear();
	if(!m_sparsescan_ids_out.empty())
		m_sparsescan_ids_out.clear();
	m_sparseindices_out.resize(miniter*maxNpos);
	m_sparsescan_ids_out.resize(miniter*maxNpos);

	for(int i=0; i<miniter; i++)
	{
		std::vector<int> indices_0;
		for (int j=0; j<diffcount; j++)
			indices_0.push_back(j);
		std::random_shuffle ( indices_0.begin(), indices_0.end() );

///////////////// test with matlab
//		char* filename="/data2/JunjingData/indices.csv";
//		std::vector<int> vecFile;
//		loadTestIGroup(filename, vecFile);
//		indices_0=vecFile;
////////////////////////

		std::vector< std::vector<int> > indices (maxNpos, std::vector<int>() ); //indices_out

		std::vector<pair<int,int> > Nitems;

		for(int j=1; j<=maxNpos; j++)
		{
			int indbegin=(j-1)*grouping;
			int indend=std::min(Npos_tmp,j*grouping);
			for(int m=indbegin; m<indend; m++)
				indices[j-1].push_back(indices_0[m]);
			int lengthNitem=indend-indbegin;
			Nitems.push_back(make_pair (-lengthNitem,j-1));
		}

//		 if ( (diffcount>1e3)||(grouping==1) )
//		 {
			std::vector< int > scan_ids (maxNpos, 1);

//		 }
		std::vector< std::vector<int> > scan_ids_out (maxNpos, std::vector<int>() );
		for(int ii=1; ii<=maxNpos; ii++)
		{
			scan_ids_out[ii-1]=std::vector<int>(indices[ii-1].size(), 1);
		}

		std::sort(Nitems.begin(), Nitems.end());

//		m_all_sparseindices_out[i].resize(Nitems.size());
//		m_all_sparsescan_ids_out[i].resize(Nitems.size());
		for(int nindex=0; nindex<Nitems.size(); nindex++)
		{
			int newindex=Nitems[nindex].second;
			m_sparseindices_out[i*Nitems.size()+nindex]=indices[newindex];
			m_sparsescan_ids_out[i*Nitems.size()+nindex]=scan_ids_out[newindex];
//			m_sparseindices_out.push_back(indices[newindex]);
//			m_sparsescan_ids_out.push_back(scan_ids_out[newindex]);
//			m_all_sparseindices_out[i][nindex]=indices[newindex];
//			m_all_sparsescan_ids_out[i][nindex]=scan_ids_out[newindex];
		}

//		m_all_sparseindices_out[i]=m_sparseindices_out;
//		m_all_sparsescan_ids_out[i]=m_sparsescan_ids_out;
	}

}



