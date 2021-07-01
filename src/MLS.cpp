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

#include "MLS.h"
#include "Cuda2DArray.hpp"
#include "utilities.h"
#include "Sample.h"
#include "Probe.h"
#include "Diffractions.h"
#include "CXMath.h"

#include <algorithm>
#include <cfloat>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MLS::MLS():m_psi(0), m_psi_old(0)
{}

MLS::~MLS()
{
	if (m_psi)		delete m_psi;
	if (m_psi_old)	delete m_psi_old;

	if(dPO) delete dPO;
	if(AA1) delete AA1;
	if(AA2) delete AA2;
	if(AA4) delete AA4;
	if(Atb1) delete Atb1;
	if(Atb2) delete Atb2;
}

void MLS::initMem(IPtychoScanMesh* scanMesh, uint2 probeSize)
{
	int regularSize=scanMesh->m_regularSize;

	dPO=new Cuda3DArray<complex_t>(regularSize, probeSize);
	AA1=new Cuda3DArray<real_t>(regularSize, probeSize);
	AA2=new Cuda3DArray<complex_t>(regularSize, probeSize);
	AA4=new Cuda3DArray<real_t>(regularSize, probeSize);
	Atb1=new Cuda3DArray<real_t>(regularSize, probeSize);
	Atb2=new Cuda3DArray<real_t>(regularSize, probeSize);
}

real_t MLS::iteration(Diffractions* diffs, Probe* probe,
						Sample* object, IPtychoScanMesh* scanMesh, std::vector< std::vector<real_t> >& fourierErrors, const std::vector<float2>& scanPositions,
						bool phaseConstraint, bool updateProbe, bool updateProbeModes, unsigned int iter, bool RMS)
{

	object->printObject(165, 161);

	const ReconstructionParams* rParams = CXParams::getInstance()->getReconstructionParams();
	int diffSize=diffs->getPatterns()->getNum();
	unsigned int iterC=iter-1;
	fourierErrors[iterC].resize(diffSize);

	if( iter==1 || (iter >= rParams->probe_pos_search) )
	{
		scanMesh->find_reconstruction_ROI_O();
	}

	uint2 objectx=scanMesh->getobject_ROIx();
	uint2 objecty=scanMesh->getobject_ROIy();
	probe->calc_object_norm(objectx, objecty, object->getObjectArray());

//	object->printObject(165, 161);

	scanMesh->updateScanPositionsDiff();

	if(rParams->variable_probe_modes && iter>rParams->probe_reconstruct)
	{
		probe->remove_extra_degree();
	}

	probe->updated_cached_illumination(scanMesh->m_oROI1, scanMesh->m_oROI2);

	int randNoIndex=rand_gen_no(1, scanMesh->m_miniter)-1;

	int group_size=scanMesh->m_sparseindices_out[0].size();
	int Nind=scanMesh->m_maxNpos;
	std::vector<int> ind_range;
	for (int i=0; i<Nind; i++)
		ind_range.push_back(i);
	std::random_shuffle (ind_range.begin(), ind_range.end());

	///////////////////////////test with matlab
	// For testing, set randNoIndex to 23 always
//	randNoIndex=3;
//	char* filename="/data2/JunjingData/ind_range.csv";
//	std::vector<int> vecFile;
//	scanMesh->loadTestIGroup(filename, vecFile);
//	ind_range=vecFile;
	///////////////////////////

	uint2 probeSize=probe->getModes()->getDimensions();
	Cuda3DArray<complex_t>* obj_proj=new Cuda3DArray<complex_t>(group_size, probeSize);
	Cuda3DArray<complex_t>* varProbe=new Cuda3DArray<complex_t>(group_size, probeSize);
	Cuda3DArray<complex_t>* tmpArr=new Cuda3DArray<complex_t>(group_size, probeSize);

	std::vector < Cuda3DArray<complex_t>* > psivec;
	Cuda3DArray<real_t>* apsi=new Cuda3DArray<real_t>(group_size, probeSize);
	Cuda3DArray<complex_t>* object_update_proj=new Cuda3DArray<complex_t>(group_size, probeSize);
	for(int i=0; i<probe->getModes()->getNum(); i++)
	{
		psivec.push_back(new Cuda3DArray<complex_t>(group_size, probeSize));
	}

	Cuda3DArray<complex_t>* probe_update=new Cuda3DArray<complex_t>(group_size, probeSize);
	CudaSmartPtr probe_update_m=new Cuda2DArray<complex_t>(probeSize.x, probeSize.y);

	CudaSmartPtr weight_proj = new Cuda2DArray<real_t>(probe->p_object->getX(), probe->p_object->getY());
	Cuda3DArray<real_t>* weight_proj_proj=new Cuda3DArray<real_t>(group_size, probeSize);
	CudaSmartPtr object_upd_sum=new Cuda2DArray<complex_t>(probe->p_object->getX(), probe->p_object->getY());
	CudaSmartPtr object_upd_precond=new Cuda2DArray<complex_t>(probe->p_object->getX(), probe->p_object->getY());
	for(int i=0; i<Nind; i++)
	{
//		object->printObject(165, 161);

		probe_update_m->set();
		weight_proj->set();
		object_upd_sum->set();
		object_upd_precond->set();
		probe_update->setToZeroes();
		weight_proj_proj->setToZeroes();
		object_update_proj->setToZeroes();
		apsi->setToZeroes();
//		varProbe->setToZeroes();

		int jj=ind_range[i];
		std::vector<int> g_ind_vec=scanMesh->m_sparseindices_out[randNoIndex*Nind+jj];
		std::vector<int> scan_idsvec=scanMesh->m_sparsescan_ids_out[randNoIndex*Nind+jj];

		if(g_ind_vec.size()!=group_size)
		{
			delete obj_proj;
			delete varProbe;
			delete tmpArr;
			delete apsi;
			delete object_update_proj;
			for(int psiindex=0; psiindex<probe->getModes()->getNum(); psiindex++)
			{
				delete psivec[psiindex];
			}
			delete probe_update;
			delete weight_proj_proj;
			psivec.clear();
			obj_proj=new Cuda3DArray<complex_t>(g_ind_vec.size(), probeSize);
			varProbe=new Cuda3DArray<complex_t>(g_ind_vec.size(), probeSize);
			tmpArr=new Cuda3DArray<complex_t>(g_ind_vec.size(), probeSize);
			apsi=new Cuda3DArray<real_t>(g_ind_vec.size(), probeSize);
			object_update_proj=new Cuda3DArray<complex_t>(g_ind_vec.size(), probeSize);
			for(int psiindex=0; psiindex<probe->getModes()->getNum(); psiindex++)
			{
				psivec.push_back(new Cuda3DArray<complex_t>(g_ind_vec.size(), probeSize));
			}
			probe_update=new Cuda3DArray<complex_t>(g_ind_vec.size(), probeSize);
			weight_proj_proj=new Cuda3DArray<real_t>(g_ind_vec.size(), probeSize);
		}
		else
		{
			if(obj_proj->getNum()!=group_size)
			{
				delete obj_proj;
				delete varProbe;
				delete tmpArr;
				delete apsi;
				delete object_update_proj;
				for(int psiindex=0; psiindex<probe->getModes()->getNum(); psiindex++)
				{
					delete psivec[psiindex];
				}
				delete probe_update;
				delete weight_proj_proj;
				psivec.clear();
				obj_proj=new Cuda3DArray<complex_t>(group_size, probeSize);
				varProbe=new Cuda3DArray<complex_t>(group_size, probeSize);
				tmpArr=new Cuda3DArray<complex_t>(group_size, probeSize);
				apsi=new Cuda3DArray<real_t>(group_size, probeSize);
				object_update_proj=new Cuda3DArray<complex_t>(group_size, probeSize);
				for(int psiindex=0; psiindex<probe->getModes()->getNum(); psiindex++)
				{
					psivec.push_back(new Cuda3DArray<complex_t>(group_size, probeSize));
				}
				probe_update=new Cuda3DArray<complex_t>(group_size, probeSize);
				weight_proj_proj=new Cuda3DArray<real_t>(group_size, probeSize);
			}
		}


//		probe->get_projections(object->getObjectArray(), obj_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);

		if(obj_proj->getNum()==1)
		{
			probe->get_projections_cpu(object->getObjectArray(), obj_proj, g_ind_vec, scanMesh->m_oROI_vec1, scanMesh->m_oROI_vec2);

		}
		else
		{	// run on GPU
			probe->get_projections(object->getObjectArray(), obj_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);
		}

//		object->printObject(165, 161);
		probe->get_illumination_probe(g_ind_vec, scanMesh->m_sub_px_shift, varProbe, psivec, obj_proj, apsi);

		if(iter/(min(20.0, pow(2, floor(2+iter*1.0/50))))==0 || iter<10)
		{//get_fourier_error
			diffs->get_fourier_error(apsi, g_ind_vec, fourierErrors[iterC]);
		}
		diffs->modulus_constraint(apsi, g_ind_vec, psivec);

		unsigned int llRange=std::max(probe->getModes()->getNum(), rParams->Nobjects);
		for(unsigned int ll=1; ll<=llRange; ll++)
		{
			int object_reconstruct=((iter>=rParams->object_reconstruct)&&(rParams->apply_multimodal_update ||  ll<=rParams->Nobjects));
			int probe_reconstruct=(iter>=rParams->probe_reconstruct);

	        int llo = std::min(rParams->Nobjects, ll);
	        int llp = std::min(probe->getModes()->getNum(), ll);

			if(probe_reconstruct)
			{
				h_multiplyConju(psivec[ll-1]->getPtr()->getDevicePtr<complex_t>(), obj_proj->getPtr()->getDevicePtr<complex_t>(),
						probe_update->getPtr()->getDevicePtr<complex_t>(), probe_update->getPtr()->getX(), probe_update->getPtr()->getY(), probe_update->getPtr()->getAlignedY());

				if(ll==1&&rParams->apply_subpix_shift)
				{
					probe->shift_probe(g_ind_vec, scanMesh->m_nsub_px_shift, probe_update);
				}

				h_addFactorDivide(probe->p_object->getDevicePtr<real_t>(), weight_proj->getDevicePtr<real_t>(), probe->MAX_ILLUM*0.01,
						weight_proj->getX(), weight_proj->getY(), weight_proj->getAlignedY());

				// TODO run on CPU
//				probe->get_projections(weight_proj, weight_proj_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);

        		if(obj_proj->getNum()==1)
        		{
        			probe->get_projections_cpu(weight_proj, weight_proj_proj, g_ind_vec, scanMesh->m_oROI_vec1, scanMesh->m_oROI_vec2);

        		}
        		else
        		{	// run on GPU
        			probe->get_projections(weight_proj, weight_proj_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);
        		}

				h_multiply(weight_proj_proj->getPtr()->getDevicePtr<real_t>(), probe_update->getPtr()->getDevicePtr<complex_t>(), tmpArr->getPtr()->getDevicePtr<complex_t>(),
						probe_update->getPtr()->getX(), probe_update->getPtr()->getY(), probe_update->getPtr()->getAlignedY());
				h_realModalSum(tmpArr->getPtr()->getDevicePtr<complex_t>(), probe_update_m->getDevicePtr<complex_t>(), tmpArr->getNum(),
						tmpArr->getDimensions().x, probe_update->getPtr()->getY(), probe_update->getPtr()->getAlignedY());
				h_normalize(probe_update_m->getDevicePtr<complex_t>(), probe_update_m->getX(), probe_update_m->getY(), probe_update_m->getAlignedY(), 1.0/tmpArr->getNum());

			}
			else
			{
				probe_update_m->set();
			}

			if(object_reconstruct)
			{
				if(llp==1)
				{
					h_multiplyConju(psivec[ll-1]->getPtr()->getDevicePtr<complex_t>(), varProbe->getPtr()->getDevicePtr<complex_t>(),
							object_update_proj->getPtr()->getDevicePtr<complex_t>(), varProbe->getPtr()->getX(), varProbe->getPtr()->getY(), varProbe->getPtr()->getAlignedY());
				}
				else
				{
					CXMath::multiply<complex_t>(probe->getModes()->getAt(llp-1), psivec[ll-1], object_update_proj);
				}

				for(int ll_tmp=1; ll_tmp<=rParams->Nobjects; ll_tmp++)
				{
					object_upd_sum->set();
					complex_t factor=make_complex_t(0, DBL_EPSILON);
					h_addFactor(object_upd_sum->getDevicePtr<complex_t>(), object_upd_sum->getDevicePtr<complex_t>(), factor,
							object_upd_sum->getX(), object_upd_sum->getY(), object_upd_sum->getAlignedY());

                    if(rParams->delta_p==0)
                    {
//                        TODO %no preconditioner as in the original ML method
//                        object_upd_sum{ll_tmp} = set_projections(object_upd_sum{ll_tmp},object_update_proj ,1, g_ind, cache);
//                        object_update_proj = get_projections(object_upd_sum{ll_tmp},object_update_proj,ll_tmp, g_ind, cache);
                    }
                    else
                    {
                    	probe->set_projections(object_upd_sum, object_update_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);

                    	h_object_sum_update_Gfun(object_upd_sum->getDevicePtr<complex_t>(), probe->p_object->getDevicePtr<real_t>(), object_upd_precond->getDevicePtr<complex_t>(),
                    			probe->MAX_ILLUM*rParams->delta_p, object_upd_sum->getX(), object_upd_sum->getY(), object_upd_sum->getAlignedY());

                    	if(rParams->beta_LSQ>0)
                    	{
                    		// Changed for CPU
//                    		probe->get_projections(object_upd_precond, object_update_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);
                    		if(obj_proj->getNum()==1)
                    		{
                    			probe->get_projections_cpu(object_upd_precond, object_update_proj, g_ind_vec, scanMesh->m_oROI_vec1, scanMesh->m_oROI_vec2);

                    		}
                    		else
                    		{	// run on GPU
                    			probe->get_projections(object_upd_precond, object_update_proj, g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2);
                    		}

                    	}

                    	if(rParams->algorithm.compare("MLs")==0)
                    	{
                    		object_upd_sum->setFromDevice<complex_t>(object_upd_precond->getDevicePtr<complex_t>(), object_upd_sum->getX(), object_upd_sum->getY());
                    	}
                    }

				}

			}
			else
			{
				object_update_proj->setToZeroes();
			}

			if(rParams->beta_LSQ>0 && ll == 1 && object_reconstruct && probe_reconstruct)
			{
				double lambda=FLT_EPSILON/(probeSize.x*probeSize.y);

				if(llp==1)
				{

					CXMath::multiply<complex_t>(probe_update_m.get(), obj_proj, dPO);
					h_get_optimal_step_lsq(psivec[ll-1]->getPtr()->getDevicePtr<complex_t>(),object_update_proj->getPtr()->getDevicePtr<complex_t>(),
							dPO->getPtr()->getDevicePtr<complex_t>(), varProbe->getPtr()->getDevicePtr<complex_t>(), lambda,
							AA1->getPtr()->getDevicePtr<real_t>(), AA2->getPtr()->getDevicePtr<complex_t>(), AA4->getPtr()->getDevicePtr<real_t>(),
							Atb1->getPtr()->getDevicePtr<real_t>(), Atb2->getPtr()->getDevicePtr<real_t>(),
							g_ind_vec.size()*dPO->getDimensions().x, dPO->getPtr()->getY(), dPO->getPtr()->getAlignedY());

					std::vector <real_t> AA1vec(g_ind_vec.size(), 0);
					std::vector <complex_t> AA2vec(g_ind_vec.size());
					std::vector <complex_t> AA3vec(g_ind_vec.size());
					std::vector <real_t> AA4vec(g_ind_vec.size(), 0);
					std::vector <real_t> Atb1vec(g_ind_vec.size(), 0);
					std::vector <real_t> Atb2vec(g_ind_vec.size(), 0);
					for(int sumIndex=0; sumIndex<g_ind_vec.size(); sumIndex++)
					{
						AA1vec[sumIndex]=h_realSum(AA1->getAt(sumIndex).getDevicePtr(), AA1->getDimensions().x, AA1->getPtr()->getY(), AA1->getPtr()->getAlignedY());
						AA2vec[sumIndex]=h_complexSum(AA2->getAt(sumIndex).getDevicePtr(), 0, AA2->getDimensions().x, 0, AA2->getPtr()->getY(), AA2->getPtr()->getAlignedY());
						AA3vec[sumIndex]=conj_complex_t(AA2vec[sumIndex]);
						AA4vec[sumIndex]=h_realSum(AA4->getAt(sumIndex).getDevicePtr(), AA4->getDimensions().x, AA4->getPtr()->getY(), AA4->getPtr()->getAlignedY());
						Atb1vec[sumIndex]=h_realSum(Atb1->getAt(sumIndex).getDevicePtr(), Atb1->getDimensions().x, Atb1->getPtr()->getY(), Atb1->getPtr()->getAlignedY());
						Atb2vec[sumIndex]=h_realSum(Atb2->getAt(sumIndex).getDevicePtr(), Atb2->getDimensions().x, Atb2->getPtr()->getY(), Atb2->getPtr()->getAlignedY());
					}

					lambda = 0.1;
					real_t meanAA1, meanAA4;
					PhaserUtil::getInstance()->mean<real_t>(AA1vec, meanAA1);
					PhaserUtil::getInstance()->mean<real_t>(AA4vec, meanAA4);
					meanAA1=lambda*meanAA1;
					meanAA4=lambda*meanAA4;

					Matrix2cf AA= Eigen::Matrix2cf(2, 2);
					Vector2cf Atb= Eigen::Vector2cf(2);
					std::vector <Vector2cf> LSQ_stepVec(g_ind_vec.size());
					LSQ_stepVec.clear();
					for(int aaindex=0; aaindex<AA1vec.size(); aaindex++)
					{
						std::complex<float> I1((AA1vec[aaindex]+meanAA1), 0);
						std::complex<float> I2(AA2vec[aaindex].x, AA2vec[aaindex].y);
						std::complex<float> I3(AA3vec[aaindex].x, AA3vec[aaindex].y);
						std::complex<float> I4((AA4vec[aaindex]+meanAA4), 0);
						AA(0,0) = I1;
						AA(0,1) = I2;
						AA(1,0) = I3;
						AA(1,1) = I4;
						Atb(0)=Atb1vec[aaindex];
						Atb(1)=Atb2vec[aaindex];
						Vector2cf xx = AA.colPivHouseholderQr().solve(Atb);
						LSQ_stepVec[aaindex]=xx;
					}

	                complex_t tempprobelsq=make_complex_t(rParams->beta_probe*rParams->beta_LSQ, 0);
	                complex_t tempobjectlsq=make_complex_t(rParams->beta_object*rParams->beta_LSQ, 0);

	                for(int indindex=0; indindex<g_ind_vec.size(); indindex++)
	                {
	            		probe->beta_probevec[g_ind_vec[indindex]]=mul_complex_t(tempprobelsq,
	            				make_complex_t(std::real(LSQ_stepVec[indindex](1)), std::imag(LSQ_stepVec[indindex](1))));
	            		object->beta_objectvec[g_ind_vec[indindex]]=mul_complex_t(tempobjectlsq,
	            				make_complex_t(std::real(LSQ_stepVec[indindex](0)), std::imag(LSQ_stepVec[indindex](0))));
	                }

				}
				else
				{
					// Since llp is always 1 when ll=1, this part never happens
//					h_get_optimal_step_lsq(psivec[ll-1]->getPtr()->getDevicePtr<complex_t>(),object_update_proj->getPtr()->getDevicePtr<complex_t>(),
//							probe_update_m->getDevicePtr<complex_t>(), obj_proj->getPtr()->getDevicePtr<complex_t>(), probe->getModes()->getAt(llp-1),
//							lambda);
				}

			}
			else if(rParams->beta_LSQ == 0 &&  ll == 1)
			{
//                cache.beta_probe(g_ind)  = par.beta_probe;
//                cache.beta_object(g_ind)  = par.beta_object;
			}
			else if(ll == 1)
			{// Looks this part will never be able to get executed, need to double comform
                /*% computationally cheaper method that assumes only
                % diagonall terms of the AA matrix
                [cache.beta_probe(g_ind),cache.beta_object(g_ind)] = ...
                    gradient_projection_solver(self,chi{ll},obj_proj{min(end,ll)},probe{ll},...
                        object_update_proj, m_probe_update,g_ind, par, cache);
                cache.beta_probe(g_ind) =   (par.beta_probe *par.beta_LSQ)*cache.beta_probe(g_ind);
                cache.beta_object(g_ind) =  (par.beta_object*par.beta_LSQ)*cache.beta_object(g_ind); */
			}

			if(object_reconstruct && (rParams->algorithm.compare("MLs")==0))
			{
				object->update_object(object_upd_sum, llo, g_ind_vec, scan_idsvec, probe->p_object, probe->MAX_ILLUM);
			}
//			object->printObject(165, 161);
			if(probe_reconstruct && ll<=probe->getModes()->getNum())
			{
				probe->update_probe(ll, probe_update_m, g_ind_vec);
			}

			// Starts when interations larget than PPS numbers
			if(iter>=rParams->probe_pos_search && ll==1)
			{
				probe->gradient_position_solver(psivec[0], obj_proj, varProbe, g_ind_vec, scanMesh->m_positions_o, scanMesh->m_positions);
			}

			// Startes at 3rd interation usually
			if((rParams->variable_probe || rParams->variable_intensity) && iter>(rParams->probe_reconstruct+1) && ll==1)
			{
				probe->update_variable_probe(probe_update_m, probe_update, obj_proj, psivec[ll-1], g_ind_vec, scanMesh->m_oROI1, scanMesh->m_oROI2,
						scanMesh->m_oROI_vec1, scanMesh->m_oROI_vec2);
			}
		}

//		object->printObject(165, 161);
//		printf("the current iteration is %d \n", i);
	}

	delete obj_proj;
	delete varProbe;
	delete apsi;
	delete probe_update;
	delete weight_proj_proj;
	delete tmpArr;
	delete object_update_proj;
	for(int i=0; i<probe->getModes()->getNum(); i++)
	{
		delete psivec[i];
	}

	if(rParams->probeModes>rParams->Nrec)
	{
		probe->ortho_modes();
	}

//	object->printObject(165, 161);
	return 1;
}

void MLS::endPhasing()
{

}
