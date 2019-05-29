########################################################################################################################
#Copyright © 2019, UChicago Argonne, LLC
#
#All Rights Reserved
#
#Software Name: ptychopy
#
#By: Argonne National Laboratory
#
#OPEN SOURCE LICENSE
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
#following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#disclaimer in the documentation and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
#derived from this software without specific prior written permission.
#
#DISCLAIMER
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
########################################################################################################################


__author__ = 'kyue'


import sys
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PIL import ImageQt

import scipy
import logging
import glob
import h5py
import logging.handlers
import scipy.constants as spc
import scipy.ndimage as spn
from readMDA import *
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

from multiprocessing import Process, Queue, Pipe
from threading import Thread

from PIL import Image
from PyQt5 import QtCore, QtGui
from scipy import misc
import time
import ptychopy


para_list=['jobID', 'fs', 'beamSize', 'qxy', 'scanDims', 'step', 'probeModes', 'rotate90',
           'sqrtData', 'fftShiftData', 'threshold', 'size', 'objectArray', 'energy', 'dx_d',
           'z', 'i','simulate', 'updateVis', 'blind', 'fp']

test_list=['gentest100', '1501', '400e-9', '153,243',   '55,75', '200e-9,200e-9', '100',  '0',
          '1', '1', '1', '256', '1600', '10.24', '172e-6',
          '4.2', '50', '', '5', '1', '/data/id2data/scan060/scan060_#06d.h5']

# test_list=['gentest100', '1501', '400e-9', '153,243',   '55,75', '200e-9,200e-9', '3',  '0',
#             '1', '1', '1', '256', '1600', '102.47933884297521', '172e-6',
#             '4.2', '50', '', '5', '1', '/mnt/xfm0-data2/data/ptycho_data/2idd/commission/h5/scan060/scan060_#06d.h5']

# columnNo=3

class Emitter(QtCore.QObject, Thread):

    signaturechanged = pyqtSignal(object, object)

    def __init__(self, transport, parent=None):
        QtCore.QObject.__init__(self, parent)
        Thread.__init__(self)
        self.transport = transport

    def _emit(self, res, fname):

        self.signaturechanged.emit(res, fname)

    def run(self):
        while True:
            try:
                signature = self.transport.recv()
            except EOFError:
                break
            else:
                self._emit(*signature)

class ChildProc(Process):

    def __init__(self, transport, queue, daemon=True):
        Process.__init__(self)
        self.transport = transport
        self.data_from_mother = queue
        self.daemon = daemon

    def initPara(self, cmdlists, cmdstrs, jobIDs, output_dir, emailstr='', ePIEdir='', its=1, visit=1):
        self.cmdlists = cmdlists
        self.cmdstrs = cmdstrs
        self.jobIDs = jobIDs
        self.output_dir = output_dir
        self.emailstr = emailstr
        self.ePIEdir = ePIEdir
        self.its = its
        self.visit = visit

    def run(self):
        while True:
            list = self.data_from_mother.get()
            self.initPara(*list)

            for cmdstr, cmdlist, jobID in zip(self.cmdstrs, self.cmdlists, self.jobIDs):

                fname = self.ePIEdir+'/data/{:s}_object_0.csv'.format(jobID)

                ptychopy.epieinit(cmdstr)

                for i in range(self.its):
                    ptychopy.epiestep()
                    if(i%self.visit==0):
                        resobj=ptychopy.epieresobj()
                        signature=(resobj, fname)
                        self.transport.send(signature)

                        # self.cmd_finished.emit(ptychopy.epieres(), fname)

                ptychopy.epiepost()


                # probe = self.opencsv(self.ePIEdir+'/data/{:s}_probes_0.csv'.format(jobID))

                # self.cmd_finished.emit(ob, fname)
                # self.cmd_finished.emit(res, fname)


                ob=self.opencsv(self.ePIEdir+'/data/{:s}_object_0.csv'.format(jobID))

                # After the reconstruction and save the image to tiff image
                angleob=np.angle(ob)
                im=Image.fromarray(angleob)
                im.save(self.output_dir+'/{:s}_ph.tif'.format(jobID))
                absob=np.abs(ob)
                im=Image.fromarray(absob)
                im.save(self.output_dir+'/{:s}_abs.tif'.format(jobID))

                # im=Image.fromarray(np.abs(probe))

                #Send email
                if(self.emailstr):
                    msg = MIMEMultipart()

                    sender=self.emailstr
                    receiver=self.emailstr

                    fname = self.output_dir+'/{:s}_ph.tif'.format(jobID)
                    arr = open(fname, 'rb').read()
                    pngFname = os.path.basename(fname).split('.')[0]+".png"

                    msg['Subject']='Reconstruction Status'
                    msg['From']=sender
                    msg['To']=receiver

                    text=MIMEText('Scan'+jobID+'is reconstructed!'+'\n'+cmdstr)
                    msg.attach(text)

                    emailImg = MIMEImage(arr, name=pngFname, _subtype="png" )
                    msg.attach(emailImg)


                    s=smtplib.SMTP('localhost')
                    s.sendmail(sender,[receiver],msg.as_string())
                    s.quit()

    def opencsv(self, stringname):
        ob=np.genfromtxt(stringname,delimiter=',',dtype=complex)
        return ob

class epieDlg(QDialog):

    changed = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super(epieDlg, self).__init__(parent)

        self.columnNo = 3

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.grid = QtGui.QGridLayout()

        btn1 = QtGui.QPushButton("Add")
        btn1.clicked.connect(self.buttonClickedAdd)
        btn1.setFixedWidth(70)

        btn2 = QtGui.QPushButton("Delete")
        btn2.clicked.connect(self.buttonClickedDelete)
        btn2.setFixedWidth(70)

        btn3 = QtGui.QPushButton("Apply")
        btn3.clicked.connect(self.apply)
        btn3.setFixedWidth(70)

        btn4 = QtGui.QPushButton("Test")
        btn4.clicked.connect(self.buttonClickedTest)
        btn4.setFixedWidth(70)

        btn5 = QtGui.QPushButton("Cancel")
        btn5.clicked.connect(self.reject)
        btn5.setFixedWidth(70)

        lb = QtGui.QLabel("GPUNo")
        le = QtGui.QLineEdit()

        self.grid.addWidget(btn1, 0, 0)
        self.grid.addWidget(btn2, 0, 1)
        self.grid.addWidget(btn3, 0, 2)
        self.grid.addWidget(btn4, 0, 3)

        self.grid.addWidget(lb, 0, 4)
        self.grid.addWidget(le, 0, 5)

        self.grid.addWidget(btn5, 0, 7)



        i = 0
        for elem in para_list:

            lb = QtGui.QLabel(elem)
            le = QtGui.QLineEdit()
            if(elem == 'fp'):
                le.setFixedWidth(200)
            else:
                le.setFixedWidth(70)

            self.grid.addWidget(lb, 1, i)
            self.grid.addWidget(le, 2, i)
            i =  i + 1

        self.setLayout(self.grid)

    def buttonClickedAdd(self):
        i = 0
        preColumn = self.columnNo - 1

        for j in range(0, len(para_list)):
            le = QtGui.QLineEdit()
            # le.setFixedWidth(70)
            preWid = self.grid.itemAtPosition(preColumn, j)
            preEdit = preWid.widget()
            le.setText(preEdit.text())

            self.grid.addWidget(le, self.columnNo, i)
            i =  i + 1

        self.columnNo = self.columnNo +1
        # print columnNo

    def buttonClickedDelete(self):
        i = 0
        # global columnNo

        if self.columnNo == 3:
            return

        curColumn = self.columnNo - 1

        for j in range(0, len(para_list)):
            curWid = self.grid.itemAtPosition(curColumn, j)
            curEdit = curWid.widget()
            # grid.removeWidget(curEdit)
            curEdit.deleteLater()
        self.columnNo = self.columnNo - 1

    def buttonClickedTest(self):
        for j in range(0, len(para_list)):
            e = self.grid.itemAtPosition(2,j)
            le = e.widget()
            le.setText(test_list[j])

    def opencsv(self, stringname):
        ob=np.genfromtxt(stringname,delimiter=',',dtype=complex)
        return ob

    def apply(self):

        # setGPUstr = "export CUDA_VISIBLE_DEVICES="
        e = self.grid.itemAtPosition(0,5)
        le = e.widget()
        gpuNo = le.text()
        # setGPUstr+=gpuNo

        # os.system(str(setGPUstr))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuNo)

        #os.chdir('/local/kyue/anlproject/ptography/ptycholib/src') # the folder where the ePIE at
        os.chdir('/home/beams/USER2IDE/ptycholib/src')

        cmdstrs = []
        jobIDs = []

        for i in range(2, self.columnNo):
            #cmdstr='/local/kyue/anlproject/ptography/ptycholib/src/ePIE' # command for ptycholib
            cmdstr='/home/beams/USER2IDE/ptycholib/src/ePIE'

            e = self.grid.itemAtPosition(i,0)
            le = e.widget()
            jobID = le.text()

            for j in range(0, len(para_list)):
                e = self.grid.itemAtPosition(i,j)
                le = e.widget()
                if para_list[j] == "sqrtData" and le.text() == '1':
                   cmdstr+=' -sqrtData'
                elif para_list[j] == "fftShiftData" and le.text() == '1':
                   cmdstr+=' -fftShiftData'
                elif para_list[j] == "energy" and le.text():
                    cmdstr+=' -'+'lambda'+'='+str(1.24/float(le.text())*10e-9) # lambda is calculated using energy
                elif le.text():
                    cmdstr+=' -'+para_list[j]+'='+le.text()
                # if le.text():
                #     cmdstr+=' -'+para_list[j]+'='+le.text()

            jobIDs.append(jobID)
            cmdstrs.append(cmdstr)

        self.changed.emit(cmdstrs, jobIDs)


class paraDlg(QDialog):

    flipScanAxisChangeSignal = pyqtSignal(object)
    mirror1stScanAxisChangeSignal = pyqtSignal(object)
    mirror2ndScanAxisChangeSignal = pyqtSignal(object)
    jitterRadiusChangeSignal = pyqtSignal(object)
    stepyChangeSignal = pyqtSignal(object)
    stepxChangeSignal = pyqtSignal(object)
    emailstrChangeSignal = pyqtSignal(object)
    hdf5pathChangeSignal = pyqtSignal(object)

    bitDepthChangeSignal = pyqtSignal(object)
    dpfChangeSignal = pyqtSignal(object)
    rot90ChangeSignal = pyqtSignal(object)

    overlapChangeSignal = pyqtSignal(object)
    shareFrequencyChangeSignal = pyqtSignal(object)

    RMSChangeSignal = pyqtSignal(object)
    updateProbeChangeSignal = pyqtSignal(object)
    updateModesChangeSignal = pyqtSignal(object)
    phaseConstraintChangeSignal = pyqtSignal(object)


    def __init__(self, parent=None):
        super(paraDlg, self).__init__(parent)

        # self.columnNo = 3

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.grid = QGridLayout()
        self.grid.addWidget(self.createGroupOne(), 0, 0)
        self.grid.addWidget(self.createGroupTwo(), 1, 0)
        self.grid.addWidget(self.createGroupThree(), 2, 0)
        self.grid.addWidget(self.createGroupFour(), 3, 0)

        self.setLayout(self.grid)

    def createGroupOne(self):
        groupBox = QGroupBox("Scan Pattern")

        widgetWidth=150
        grid = QGridLayout()

        self.flipScanAxis=0
        self.flipScanAxislb = QLabel("flipScanAxis:")
        # self.flipScanAxislb.setFixedWidth(90)
        self.flipScanAxisCbox = QComboBox()
        self.flipScanAxisCbox.setToolTip('flipScanAxis')
        self.flipScanAxisCbox.addItems(
                    ["{0}".format(x) for x in range(0,2)]
        )
        self.flipScanAxisCbox.setFixedWidth(widgetWidth)
        self.flipScanAxisCbox.currentIndexChanged.connect(self.flipScanAxisChanged)

        self.mirror1stScanAxis=0
        self.mirror1stScanAxislb = QLabel("mirror1stScanAxis:")
        # self.mirror1stScanAxislb.setFixedWidth(90)
        # self.distancelb.setFixedWidth(90)
        self.mirror1stScanAxisCbox = QComboBox()
        self.mirror1stScanAxisCbox.setToolTip('mirror1stScanAxis')
        self.mirror1stScanAxisCbox.addItems(
                    ["{0}".format(x) for x in range(0,2)]
        )
        self.mirror1stScanAxisCbox.setFixedWidth(widgetWidth)
        self.mirror1stScanAxisCbox.currentIndexChanged.connect(self.mirror1stScanAxisChanged)

        self.mirror2ndScanAxis=0
        self.mirror2ndScanAxislb = QLabel("mirror2ndScanAxis:")
        # self.mirror2ndScanAxislb.setFixedWidth(90)
        # self.distancelb.setFixedWidth(90)
        self.mirror2ndScanAxisCbox = QComboBox()
        self.mirror2ndScanAxisCbox.setToolTip('mirror2ndScanAxis')
        self.mirror2ndScanAxisCbox.addItems(
                    ["{0}".format(x) for x in range(0,2)]
        )
        self.mirror2ndScanAxisCbox.setFixedWidth(widgetWidth)
        self.mirror2ndScanAxisCbox.currentIndexChanged.connect(self.mirror2ndScanAxisChanged)

        self.jitterRadius=0
        self.jitterRadiuslb = QLabel("jitterRadius:")
        self.jitterRadiusle = QLineEdit()
        self.jitterRadiusle.setToolTip('jitterRadius')
        self.jitterRadiusle.setText(str(self.jitterRadius))
        self.jitterRadiusle.setFixedWidth(widgetWidth)
        self.jitterRadiusle.textChanged.connect(self.jitterRadiusChanged)

        self.stepy=""
        self.stepx=""
        self.step=(self.stepy,self.stepx)

        self.stepylb = QLabel("stepy:")
        self.stepyle = QLineEdit()
        self.stepyle.setToolTip('stepy')
        self.stepyle.setText(str(self.stepy))
        self.stepyle.setFixedWidth(widgetWidth)
        self.stepyle.textChanged.connect(self.stepyChanged)

        self.stepxlb = QLabel("stepx:")
        self.stepxle = QLineEdit()
        self.stepxle.setToolTip('stepx')
        self.stepxle.setText(str(self.stepx))
        self.stepxle.setFixedWidth(widgetWidth)
        self.stepxle.textChanged.connect(self.stepxChanged)

        self.emaillb = QLabel("Email Address on Completion :")
        self.emaille = QLineEdit()
        self.emaille.textChanged.connect(self.emailstrChanged)

        self.hdfpath=""
        self.hdf5pathlb = QLabel("HDF 5 Path name:")
        self.hdf5pathle = QLineEdit()
        self.hdf5pathle.setText(str(self.hdfpath))
        self.hdf5pathle.setFixedWidth(widgetWidth)
        self.hdf5pathle.textChanged.connect(self.hdf5pathChanged)

        grid.addWidget(self.flipScanAxislb, 0, 0)
        grid.addWidget(self.flipScanAxisCbox, 0, 1)
        grid.addWidget(self.mirror1stScanAxislb, 1, 0)
        grid.addWidget(self.mirror1stScanAxisCbox, 1, 1)
        grid.addWidget(self.mirror2ndScanAxislb, 2, 0)
        grid.addWidget(self.mirror2ndScanAxisCbox, 2, 1)
        grid.addWidget(self.jitterRadiuslb, 3, 0)
        grid.addWidget(self.jitterRadiusle, 3, 1)
        grid.addWidget(self.stepylb, 4, 0)
        grid.addWidget(self.stepyle, 4, 1)
        grid.addWidget(self.stepxlb, 5, 0)
        grid.addWidget(self.stepxle, 5, 1)
        grid.addWidget(self.emaillb, 6, 0)
        grid.addWidget(self.emaille, 6, 1)
        grid.addWidget(self.hdf5pathlb, 7, 0)
        grid.addWidget(self.hdf5pathle, 7, 1)

        groupBox.setLayout(grid)

        return groupBox

    def setGroupOnePara(self, flipScanAxis, mirror1stScanAxis, mirror2ndScanAxis, jitterRadius, stepy, stepx, emailstr, hdf5path):

        self.flipScanAxisCbox.setCurrentIndex(flipScanAxis)
        self.mirror1stScanAxisCbox.setCurrentIndex(mirror1stScanAxis)
        self.mirror2ndScanAxisCbox.setCurrentIndex(mirror2ndScanAxis)
        self.jitterRadiusle.setText(str(jitterRadius))
        self.stepyle.setText(str(stepy))
        self.stepxle.setText(str(stepx))
        self.emaille.setText(str(emailstr))
        self.hdf5pathle.setText(str(hdf5path))

    def flipScanAxisChanged(self):
        self.flipScanAxisChangeSignal.emit(int(self.flipScanAxisCbox.currentIndex()))

    def mirror1stScanAxisChanged(self):
        self.mirror1stScanAxisChangeSignal.emit(int(self.mirror1stScanAxisCbox.currentIndex()))

    def mirror2ndScanAxisChanged(self):
        self.mirror2ndScanAxisChangeSignal.emit(int(self.mirror2ndScanAxisCbox.currentIndex()))

    def jitterRadiusChanged(self):
        if(self.jitterRadiusle.text()):
            self.jitterRadiusChangeSignal.emit(int(self.jitterRadiusle.text()))

    def stepyChanged(self):
        if(self.stepyle.text()):
            self.stepyChangeSignal.emit(str(self.stepyle.text()))

    def stepxChanged(self):
        if(self.stepxle.text()):
            self.stepxChangeSignal.emit(str(self.stepxle.text()))

    def emailstrChanged(self):
        if(self.emaille.text()):
            self.emailstrChangeSignal.emit(str(self.emaille.text()))

    def hdf5pathChanged(self):
        if(self.hdf5pathle.text()):
            self.hdf5pathChangeSignal.emit(str(self.hdf5pathle.text()))

    def createGroupTwo(self):
        groupBox = QGroupBox("Diff Pattern")

        widgetWidth=150
        grid = QGridLayout()

        self.bitDepth=32
        self.bitDepthlb = QLabel("bitDepth:")
        self.bitDepthle = QLineEdit()
        self.bitDepthle.setToolTip('bitDepth')
        self.bitDepthle.setText(str(self.bitDepth))
        self.bitDepthle.setFixedWidth(widgetWidth)
        self.bitDepthle.textChanged.connect(self.bitDepthChanged)

        self.DPF=1
        self.DPFlb = QLabel("DPF:")
        self.DPFle = QLineEdit()
        self.DPFle.setToolTip('DPF')
        self.DPFle.setText(str(self.DPF))
        self.DPFle.setFixedWidth(widgetWidth)
        self.DPFle.textChanged.connect(self.dpfChanged)

        self.rot90=0
        self.rotatelb = QLabel("Rotate 90:")
        # self.distancelb.setFixedWidth(90)
        self.rotateCbox = QComboBox()
        self.rotateCbox.setToolTip('rotate90')
        self.rotateCbox.addItems(
                    ["{0}".format(x) for x in range(0,4)]
        )
        self.rotateCbox.currentIndexChanged.connect(self.rot90Changed)

        grid.addWidget(self.bitDepthlb, 0, 0)
        grid.addWidget(self.bitDepthle, 0, 1)
        grid.addWidget(self.DPFlb, 1, 0)
        grid.addWidget(self.DPFle, 1, 1)
        grid.addWidget(self.rotatelb, 2, 0)
        grid.addWidget(self.rotateCbox, 2, 1)

        groupBox.setLayout(grid)

        return groupBox

    def setGroupTwoPara(self, bitDepth, dpf, rot90):
        self.bitDepthle.setText(str(bitDepth))
        self.DPFle.setText(str(dpf))
        self.rotateCbox.setCurrentIndex(rot90)

    def bitDepthChanged(self):
        if(self.bitDepthle.text()):
            self.bitDepthChangeSignal.emit(int(self.bitDepthle.text()))

    def dpfChanged(self):
        if(self.DPFle):
            self.dpfChangeSignal.emit(int(self.DPFle.text()))

    def rot90Changed(self):

        self.rot90ChangeSignal.emit(int(self.rotateCbox.currentIndex()))

    def createGroupThree(self):
        groupBox = QGroupBox("Multi GPU")

        widgetWidth=150
        grid = QGridLayout()

        self.overlap=0
        self.overlaplb = QLabel("overlap:")
        self.overlaple = QLineEdit()
        self.overlaple.setToolTip('overlap')
        self.overlaple.setText(str(self.overlap))
        self.overlaple.setFixedWidth(widgetWidth)
        self.overlaple.textChanged.connect(self.overlapChanged)

        self.shareFrequency=10
        self.shareFrequencylb = QLabel("shareFrequency:")
        self.shareFrequencyle = QLineEdit()
        self.shareFrequencyle.setToolTip('shareFrequency')
        self.shareFrequencyle.setText(str(self.shareFrequency))
        self.shareFrequencyle.setFixedWidth(widgetWidth)
        self.shareFrequencyle.textChanged.connect(self.shareFrequencyChanged)

        grid.addWidget(self.overlaplb, 0, 0)
        grid.addWidget(self.overlaple, 0, 1)
        grid.addWidget(self.shareFrequencylb, 1, 0)
        grid.addWidget(self.shareFrequencyle, 1, 1)

        groupBox.setLayout(grid)

        return groupBox

    def setGroupThreePara(self, overlap, shareFrequency):
        self.overlaple.setText(str(overlap))
        self.shareFrequencyle.setText(str(shareFrequency))

    def overlapChanged(self):
        if(self.overlaple.text()):
            self.overlapChangeSignal.emit(int(self.overlaple.text()))

    def shareFrequencyChanged(self):
        if(self.shareFrequencyle.text()):
            self.shareFrequencyChangeSignal.emit(int(self.shareFrequencyle.text()))

    def createGroupFour(self):
        groupBox = QGroupBox("Reconstruction")

        widgetWidth=150
        grid = QGridLayout()

        # default value for RMS?
        self.RMS=10
        self.RMSlb = QLabel("RMS:")
        self.RMSle = QLineEdit()
        self.RMSle.setToolTip('RMS')
        self.RMSle.setText(str(self.RMS))
        self.RMSle.setFixedWidth(widgetWidth)
        self.RMSle.textChanged.connect(self.RMSChanged)

        self.updateProbe=10
        self.updateProbelb = QLabel("updateProbe:")
        self.updateProbele = QLineEdit()
        self.updateProbele.setToolTip('updateProbe')
        self.updateProbele.setText(str(self.updateProbe))
        self.updateProbele.setFixedWidth(widgetWidth)
        self.updateProbele.textChanged.connect(self.updateProbeChanged)

        self.updateModes=20
        self.updateModeslb = QLabel("updateModes:")
        self.updateModesle = QLineEdit()
        self.updateModesle.setToolTip('updateModes')
        self.updateModesle.setText(str(self.updateModes))
        self.updateModesle.setFixedWidth(widgetWidth)
        self.updateModesle.textChanged.connect(self.updateModesChanged)

        self.phaseConstraint=1
        self.phaseConstraintlb = QLabel("phaseConstraint:")
        self.phaseConstraintle = QLineEdit()
        self.phaseConstraintle.setToolTip('phaseConstraint')
        self.phaseConstraintle.setText(str(self.phaseConstraint))
        self.phaseConstraintle.setFixedWidth(widgetWidth)
        self.phaseConstraintle.textChanged.connect(self.phaseConstraintChanged)

        grid.addWidget(self.RMSlb, 0, 0)
        grid.addWidget(self.RMSle, 0, 1)
        grid.addWidget(self.updateProbelb, 1, 0)
        grid.addWidget(self.updateProbele, 1, 1)
        grid.addWidget(self.updateModeslb, 2, 0)
        grid.addWidget(self.updateModesle, 2, 1)
        grid.addWidget(self.phaseConstraintlb, 3, 0)
        grid.addWidget(self.phaseConstraintle, 3, 1)

        groupBox.setLayout(grid)

        return groupBox

    def setGroupFourPara(self, RMS, updateProbe, updateModes, phaseConstraint):
        self.RMSle.setText(str(RMS))
        self.updateProbele.setText(str(updateProbe))
        self.updateModesle.setText(str(updateModes))
        self.phaseConstraintle.setText(str(phaseConstraint))

    def RMSChanged(self):
        if(self.RMSle.text()):
            self.RMSChangeSignal.emit(int(self.RMSle.text()))

    def updateProbeChanged(self):
        if(self.updateProbele.text()):
            self.updateProbeChangeSignal.emit(int(self.updateProbele.text()))

    def updateModesChanged(self):
        if(self.updateModesle.text()):
            self.updateModesChangeSignal.emit(int(self.updateModesle.text()))

    def phaseConstraintChanged(self):
        if(self.phaseConstraintle.text()):
            self.phaseConstraintChangeSignal.emit(int(self.phaseConstraintle.text()))



class ImgWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        self.mother_pipe, self.child_pipe = Pipe()
        self.queue = Queue()
        self.data_to_child = self.queue
        self.emitter = Emitter(self.mother_pipe)
        self.emitter.daemon = True
        self.emitter.start()
        self.emitter.signaturechanged.connect(self.cmd_data_ready)
        ChildProc(self.child_pipe, self.queue).start()


        self.cmdstrs=[]
        self.jobIDs=[]
        self.filename=""

        self.scanNo=[]  #for test

        self.stxm_dpc_only=0  #1: only calcualte differentiate phase contrast, otherwise do ptychography reconstruction
        self.det_mod=0  #1: prepare data for python CXPhasing
        self.get_pos=0  #1: generating position files

        self.prefix='dec_comm'
        self.beamsize=100.0  #nm
        self.pNum=5

        self.diffSize=256 #diffraction pattern size for reconstruction
        self.pixelSize=75 #detector pixel size, um
        self.distance=1.95 #distance between sample and detector, m
        self.ts=1 #threshold, below which is set to be 0
        self.fs=1 #start file number, default 1, 2 is for the empty diffraction pattern when open the shutter
        self.its=50
        self.visit=1 # the initial iteration is the same with the overall iteration

        self.sqrt_data = 1
        self.fft_shift_data = 1
        self.flipScanAxis = 0

        self.energy = 10
        self.simulate = 0
        self.scan_dims = (26,26)
        self.step=(200e-9,200e-9)
        self.objectArray=16
        self.jobID="gentest100"
        self.qxy=(153,243)
        self.probeModes=3
        self.jitterRadius = 0
        self.shareFrequency = 0

        self.dpf = 1



        self.probeGuess = ''
        self.positionFile = ''

        # self.userdata_dir='/mnt/xfm0-data2/data/2iddf_Velo/data17c1/Apr15_2017'  #User data directory, or main directory
        # self.mda_path=os.path.join(self.userdata_dir,'mda/2iddf_{:04d}.mda')
        # self.ePIEdir='/home/beams/USER2IDE/ptycholib/src'

        # self.userdata_dir='/data1/JunjingData'  #User data directory, or main directory
        self.userdata_dir = ""
        # self.mda_path=os.path.join(self.userdata_dir,'mda/2iddf_{:04d}.mda')
        # self.log_dir=os.path.join(self.userdata_dir,'log')
        # self.logCreate()

        self.cmdlists = []


        # Non important dialog parameter start from here
        self.dialog=paraDlg(self)

        # Scan Pattern parameter
        self.flipScanAxis=0
        self.mirror1stScanAxis=0
        self.mirror2ndScanAxis=0
        self.jitterRadius=0
        self.stepy=""
        self.stepx=""
        self.emailstr =""
        self.hdf5path="/entry/data/data"

        # Diff pattern parameter
        self.bitDepth=32
        self.dpf=1
        self.rot90=0

        # Multi GPU parameter
        self.overlap=0
        self.shareFrequency=10

        # Reconstruction
        self.RMS=0
        self.updateProbe=10
        self.updateModes=20
        self.phaseConstraint=1

        self.scenel = QGraphicsScene()
        self.viewl = QGraphicsView(self.scenel)

        self.scener = QGraphicsScene()
        self.viewr = QGraphicsView(self.scener)

        self.runButton = QPushButton("Run")
        self.runButton.setFixedWidth(90)
        # self.button2 = QPushButton("Configure")
        # self.button2.setFixedWidth(90)
        # self.button3 = QPushButton("Open CSV")
        # self.button3.setFixedWidth(90)
        self.button4 = QPushButton("Open Images")
        self.button4.setFixedHeight(22)

        self.button5 = QPushButton("Set_ReDir")
        self.button5.setFixedWidth(90)

        # self.button5 = QPushButton("Load")
        # self.button5.setFixedWidth(90)
        # self.button6 = QPushButton("Save")
        # self.button6.setFixedWidth(90)

        self.gpulb = QLabel("GPU Node:")
        self.gpuCbox = QComboBox()
        self.gpuCbox.addItems(
                    ["{0}".format(x) for x in range(0,7)]
        )

        self.beamlinelb = QLabel("Beamline:")
        self.beamlineCbox = QComboBox()
        self.beamlineCbox.setFixedWidth(120)
        self.beamlineCbox.addItems(
                    ['Velociprobe','Bionanoprobe','2-ID-D']
        )

        self.distancelb = QLabel("Detector distance (m):")
        # self.distancelb.setFixedWidth(90)
        self.distancele = QLineEdit()
        self.distancele.setToolTip('z')
        self.distancele.setText(str(self.distance))
        self.distancele.setFixedWidth(90)

        self.pixelsizelb = QLabel("Pixel size(um):")
        # self.pixelsizelb.setFixedWidth(90)
        self.pixelsizele = QLineEdit()
        self.pixelsizele.setToolTip('dx_d')
        self.pixelsizele.setText(str(self.pixelSize))
        self.pixelsizele.setFixedWidth(90)

        self.diffpattlb = QLabel("Diff. pattern center (Y | X):")

        self.diffpattyle = QLineEdit()
        self.diffpattyle.setToolTip("qy")
        self.diffpattyle.setFixedWidth(90)
        self.diffpattxle = QLineEdit()
        self.diffpattxle.setToolTip("qx")
        self.diffpattxle.setFixedWidth(90)

        self.sizelb = QLabel("Diff. pattern size:")
        # self.distancelb.setFixedWidth(90)
        self.sizele = QLineEdit()
        self.sizele.setToolTip('size')
        self.sizele.setText(str(self.diffSize))
        self.sizele.setFixedWidth(90)

        self.rotatelb = QLabel("Rotate 90:")
        # self.distancelb.setFixedWidth(90)
        self.rotateCbox = QComboBox()
        self.rotateCbox.setToolTip('rotate90')
        self.rotateCbox.addItems(
                    ["{0}".format(x) for x in range(0,4)]
        )
        # self.rotatele.setFixedWidth(90)

        self.probelb = QLabel("Probe Beam size (nm):")
        # self.distancelb.setFixedWidth(90)
        self.probele = QLineEdit()
        self.probele.setToolTip('beamSize')
        self.probele.setText(str(self.beamsize))
        self.probele.setFixedWidth(90)

        self.probemlb = QLabel("Probe modes:")
        # self.distancelb.setFixedWidth(90)
        self.probemle = QLineEdit()
        self.probemle.setToolTip('probeModes')
        self.probemle.setText(str(self.pNum))
        self.probemle.setFixedWidth(90)

        self.tslb = QLabel("Threshold:")
        # self.distancelb.setFixedWidth(90)
        self.tsle = QLineEdit()
        self.tsle.setToolTip('Threshold')
        self.tsle.setText(str(self.ts))
        self.tsle.setFixedWidth(90)

        self.itslb = QLabel("Iterations:")
        # self.distancelb.setFixedWidth(90)
        self.itsle = QLineEdit()
        self.itsle.setToolTip("i")
        self.itsle.setText(str(self.its))
        self.itsle.setFixedWidth(90)

        self.fslb = QLabel("File start No.:")
        # self.distancelb.setFixedWidth(90)
        self.fsle = QLineEdit()
        self.fsle.setToolTip("fs")
        self.fsle.setText(str(self.fs))
        self.fsle.setFixedWidth(90)

        self.nylb = QLabel("Lines for recon:")
        # self.distancelb.setFixedWidth(90)
        self.nyle = QLineEdit()
        self.nyle.setToolTip("ny")
        #self.nyle.setText(str(self.ny))
        self.nyle.setFixedWidth(90)

        self.energylb = QLabel("Energy:")
        # self.distancelb.setFixedWidth(90)
        self.energyle = QLineEdit()
        self.energyle.setText(str(self.energy))
        self.energyle.setFixedWidth(90)

        self.visitlb = QLabel("Vis iteration:")
        # self.distancelb.setFixedWidth(90)
        self.visitle = QLineEdit()
        self.visitle.setToolTip("iterations for realtime visualization")
        self.visitle.setText(str(self.visit))
        self.visitle.setFixedWidth(90)

        self.prelb = QLabel("Prefix:")
        # self.distancelb.setFixedWidth(90)
        self.prele = QLineEdit()
        self.prele.setText(self.prefix)
        self.prele.setFixedWidth(90)

        # self.emaillb = QLabel("Email Address on Completion :")
        # self.emaille = QLineEdit()

        self.Tlb = QLabel("T :")
        self.Tle = QLineEdit()
        self.Tle.setFixedWidth(90)

        self.algorithm="ePIE"
        self.algorithmlb = QLabel("algorithm:")
        # self.distancelb.setFixedWidth(90)
        self.algorithmle = QLineEdit()
        self.algorithmle.setText(self.algorithm)
        self.algorithmle.setFixedWidth(90)

        self.stxmCheckBox = QCheckBox("STXM & DPC only")
        self.stxmCheckBox.setChecked(self.stxm_dpc_only)

        self.detmodCheckBox = QCheckBox("Det_mod")
        self.detmodCheckBox.setChecked(self.det_mod)

        self.getposCheckBox = QCheckBox("Get_pos")
        self.getposCheckBox.setChecked(self.get_pos)

        self.fplb = QLabel("Job Path:")
        self.fpbutton = QPushButton("Browse MDA")
        self.fpbutton.setFixedWidth(90)
        self.fple = QLineEdit()
        self.fplw = QListWidget()
        # self.fplw.setFixedWidth(270)

        self.probeGuesslb = QLabel("ProbeGuess Path:")
        self.probeGuessbutton = QPushButton("ProbeGuess")
        self.probeGuessbutton.setFixedWidth(90)
        self.probeGuessle = QLineEdit()

        self.positionFilelb = QLabel("Position File:")
        self.positionFilebutton = QPushButton("Positions")
        self.positionFilebutton.setFixedWidth(90)
        self.positionFilele = QLineEdit()

        # self.objectGuesslb = QLabel("objectGuess Path:")
        self.objectGuessbutton = QPushButton("objectGuess")
        self.objectGuessbutton.setFixedWidth(90)
        self.objectGuessle = QLineEdit()

        self.imgPathLlbl = QLabel("Image showing:")
        self.imgPathLlbr = QLabel("Image showing:")

        self.emptylb = QLabel(" ")

        self.noImportlb = QLabel("Non important parm")
        self.noImportbn = QPushButton("Parameter")

        self.ePIEdir='.'
        self.output_dir=os.path.join(self.ePIEdir,'data', self.prefix)
        self.create_dest_dirs(self.output_dir)

        self.ePIEdirbutton=QPushButton("Data Path")
        self.ePIEdirbutton.setFixedWidth(90)
        self.ePIEdirlb=QLabel(".")


        grid = QGridLayout()
        #grid.addWidget(self.button2, 0, 1)
        #grid.addWidget(self.button3, 0, 1)
        #grid.addWidget(self.button4, 0, 2)

        grid.addWidget(self.beamlinelb, 0, 0, 1, 2)
        grid.addWidget(self.beamlineCbox, 1, 0)

        grid.addWidget(self.prelb, 0, 2, 1, 1)
        grid.addWidget(self.prele, 1, 2)

        grid.addWidget(self.distancelb, 2, 0, 1, 2)
        grid.addWidget(self.distancele, 2, 2)

        grid.addWidget(self.pixelsizelb, 3, 0, 1, 2)
        grid.addWidget(self.pixelsizele, 3, 2)

        grid.addWidget(self.sizelb, 4, 0, 1, 2)
        grid.addWidget(self.sizele, 4, 2)

        grid.addWidget(self.gpulb, 5, 0, 1, 2)
        grid.addWidget(self.gpuCbox, 5, 2)

        # grid.addWidget(self.rotatelb, 6, 0, 1, 2)
        # grid.addWidget(self.rotateCbox, 6, 2)

        grid.addWidget(self.probelb, 7, 0, 1, 2)
        grid.addWidget(self.probele, 7, 2)

        grid.addWidget(self.probemlb, 8, 0, 1, 2)
        grid.addWidget(self.probemle, 8, 2)

        grid.addWidget(self.tslb, 9, 0, 1, 2)
        grid.addWidget(self.tsle, 9, 2)

        grid.addWidget(self.itslb, 10, 0, 1, 2)
        grid.addWidget(self.itsle, 10, 2)

        grid.addWidget(self.fslb, 11, 0, 1, 2)
        grid.addWidget(self.fsle, 11, 2)

        grid.addWidget(self.nylb, 12, 0, 1, 2)
        grid.addWidget(self.nyle, 12, 2)

        grid.addWidget(self.energylb, 13, 0, 1, 2)
        grid.addWidget(self.energyle, 13, 2)

        grid.addWidget(self.diffpattlb, 14, 0, 1, 3)
        grid.addWidget(self.diffpattxle, 15, 2)
        grid.addWidget(self.diffpattyle, 15, 1)

        # grid.addWidget(self.emaillb, 16, 0, 1, 3)
        # grid.addWidget(self.emaille, 17, 1, 1, 2)

        grid.addWidget(self.Tlb, 16, 0, 1, 2)
        grid.addWidget(self.Tle, 16, 2)

        grid.addWidget(self.algorithmlb, 17, 0, 1, 2)
        grid.addWidget(self.algorithmle, 17, 2)

        grid.addWidget(self.noImportlb, 18, 0)
        grid.addWidget(self.noImportbn, 18, 2)

        grid.addWidget(self.visitlb, 19, 0, 1, 2)
        grid.addWidget(self.visitle, 19, 2)

        grid.addWidget(self.emptylb)

        grid.addWidget(self.fplb, 21, 0)
        grid.addWidget(self.fpbutton, 21, 1)
        grid.addWidget(self.button5, 21, 2)
        grid.addWidget(self.fple, 22, 0, 1, 3)
        grid.addWidget(self.fplw, 23, 0, 3, 3)

        # grid.addWidget(self.probeGuesslb, 31, 0, 1, 2)

        grid.addWidget(self.stxmCheckBox, 30, 0, 1, 1)
        grid.addWidget(self.detmodCheckBox, 30, 1, 1, 1)
        grid.addWidget(self.getposCheckBox, 30, 2, 1, 1)

        grid.addWidget(self.probeGuessbutton, 31, 0)
        grid.addWidget(self.positionFilebutton, 31, 1)
        grid.addWidget(self.runButton, 31,2 )
        grid.addWidget(self.probeGuessle, 32, 0, 1, 3)
        grid.addWidget(self.positionFilele, 33, 0, 1, 3)

        grid.addWidget(self.objectGuessbutton, 34, 0, 1, 3)
        grid.addWidget(self.objectGuessle, 35, 0, 1, 3)

        grid.addWidget(self.ePIEdirbutton, 36, 0, 1, 3)
        grid.addWidget(self.ePIEdirlb, 37, 0, 1, 3)

        grid.setAlignment(Qt.AlignTop)

        self.s1 = QScrollBar()
        self.s1.setOrientation(Qt.Horizontal)
        self.s1.setMaximum(0)
        self.s1.setFixedHeight(22)
        self.s1.sliderMoved.connect(self.sliderval)
        self.s1.valueChanged.connect(self.sliderval)



        vlayout1 = QVBoxLayout()
        vlayout1.addWidget(self.s1)
        vlayout1.addWidget(self.viewl)
        vlayout1.addWidget(self.imgPathLlbl)
        vlayout2 = QVBoxLayout()
        #vlayout2.addWidget(self.emptylb)
        vlayout2.addWidget(self.button4)
        vlayout2.addWidget(self.viewr)
        vlayout2.addWidget(self.imgPathLlbr)

        hlayout = QHBoxLayout()
        hlayout.addLayout(grid)
        hlayout.addLayout(vlayout1)
        hlayout.addLayout(vlayout2)

        self.setLayout(hlayout)

        self.runButton.clicked.connect(self.apply)

        self.noImportbn.clicked.connect(self.noImportConfig)
        # self.button2.clicked.connect(self.config)
        # self.button3.clicked.connect(self.fileOpen)

        self.button4.clicked.connect(self.batchOpen)
        self.button5.clicked.connect(self.resultDirectorySet)

        self.fpbutton.clicked.connect(self.browserMDA)
        self.probeGuessbutton.clicked.connect(self.browserProbeGuess)
        self.positionFilebutton.clicked.connect(self.browserPositionFile)
        self.objectGuessbutton.clicked.connect(self.objectGuessFile)
        self.ePIEdirbutton.clicked.connect(self.ePIEFile)

    def batchOpen(self):
        """
        Batch open the image

        """
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")

        formats = (["*.{0}".format(str("csv").lower()),"*.{0}".format(str("npy").lower())])
        self.fnames = QFileDialog.getOpenFileNames(self,
                                                    "Batch open -- Choose File", dir,
                                                    "Files ({0})".format(" ".join(formats)))[0]


        if len(self.fnames):
            # numbe = self.fnames.count()
            numbe = len(self.fnames)
            self.s1.setMaximum(numbe-1)
            self.s1.setValue(0)
            print(self.fnames[0])
            self.loadFile(self.fnames[0])

    def closeEvent(self, event):
        """
        Event for closing the window
        :param event:
        """
        self.beamline = int(self.beamlineCbox.currentIndex())
        self.settings.setValue('self_beamline', self.beamline)

        # self.stxm_dpc_only = int(self.stxmCheckBox.isChecked())
        # self.settings.setValue('self_stxm_dpc_only', self.stxm_dpc_only)
        #
        # self.det_mod = int(self.detmodCheckBox.isChecked())
        # self.settings.setValue('self_det_mod', self.det_mod)

        self.energy = float(self.energyle.text())
        self.settings.setValue('self_energy', self.energy)

        self.beamsize = float(self.probele.text())
        self.settings.setValue('self_beamsize', self.beamsize)

        self.prefix = str(self.prele.text())
        self.settings.setValue('self_prefix', self.prefix)

        self.pNum = int(self.probemle.text())
        self.settings.setValue('self_pNum', self.pNum)

        self.gpuNo = int(self.gpuCbox.currentIndex())
        self.settings.setValue('self_gpuNo', self.gpuNo)

        self.rot90 = int(self.rotateCbox.currentIndex())
        self.settings.setValue('self_rot90', self.rot90)

        self.diffSize = int(self.sizele.text())
        self.settings.setValue('self_diffSize', self.diffSize)

        self.pixelSize = int(self.pixelsizele.text())
        self.settings.setValue('self_pixelSize', self.pixelSize)

        self.distance = float(self.distancele.text())
        self.settings.setValue('self_distance', self.distance)

        self.ts = int(self.tsle.text())
        self.settings.setValue('self_ts', self.ts)

        self.fs = int(self.fsle.text())
        self.settings.setValue('self_fs', self.fs)

        self.its = int(self.itsle.text())
        self.settings.setValue('self_its', self.its)

        self.visit = int(self.visitle.text())
        self.settings.setValue('self_visit', self.visit)

        # self.emailstr = str(self.emaille.text())
        # self.settings.setValue('self_emailstr', self.emailstr)

        del self.settings

    def runEPIE(self):
        """
        run the epie library in the thread

        """
        gpuNo = self.gpuCbox.currentIndex()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuNo)

        # self.threads = []
        # worker = Worker(self.cmdlists, self.cmdstrs, self.jobIDs, self.output_dir, self.emailstr, \
        #                    self.ePIEdir, self.its, self.visit)
        # thread = QThread()
        # self.threads.append((thread, worker))
        # worker.moveToThread(thread)
        #
        # worker.cmd_finished.connect(self.cmd_data_ready)
        #
        # thread.started.connect(worker.work)
        # thread.start()

        list = (self.cmdlists, self.cmdstrs, self.jobIDs, self.output_dir, self.emailstr, \
                           self.ePIEdir, self.its, self.visit)
        self.data_to_child.put(list)

    def cmd_data_ready(self, ob, fname=""):
        """
        when the data is ready
        :param ob:
        :param fname:
        """

        angleob=np.angle(ob)
        absob=np.abs(ob)

        # self.log.info('Saving images to: '+self.output_dir)
        # ph = np.concatenate((angleob, absob), axis=1)

        imgl = scipy.misc.toimage(np.flipud(angleob))
        self.display_image_l(imgl)
        self.imgPathLlbl.setText("Recon: "+fname)

        imgr = scipy.misc.toimage(np.flipud(absob))
        self.display_image_r(imgr)
        self.imgPathLlbr.setText("Recon: "+fname)

        QCoreApplication.processEvents()
        time.sleep(0.5)

    def dpc_data_ready(self, obl, obr):
        """
        when dpc dat is ready
        :param obl: left image data
        :param obr: right image data
        """
        imgl = scipy.misc.toimage(np.flipud(obl))
        self.display_image_l(imgl)

        imgr = scipy.misc.toimage(np.flipud(obr))
        self.display_image_r(imgr)


        QCoreApplication.processEvents()
        time.sleep(0.5)


    def display_image_l(self, img):
        """
        Display the left image
        :param img:
        """
        self.scenel.clear()
        w, h = img.size
        self.imgQ = ImageQt.ImageQt(img)
        pixMap = QPixmap.fromImage(self.imgQ)
        self.scenel.addPixmap(pixMap)
        self.viewl.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
        self.scenel.update()

    def display_image_r(self, img):
        """
        Display the right image
        :param img:
        """
        self.scener.clear()
        w, h = img.size
        self.imgQ = ImageQt.ImageQt(img)
        pixMap = QPixmap.fromImage(self.imgQ)
        self.scener.addPixmap(pixMap)
        self.viewr.fitInView(QRectF(0, 0, w, h), Qt.KeepAspectRatio)
        self.scener.update()

    def noImportConfig(self):
        """
        Show the dialog with all the non impor

        """
        self.dialog = paraDlg(self)
        self.dialog.setGroupOnePara(self.flipScanAxis, self.mirror1stScanAxis, self.mirror2ndScanAxis, self.jitterRadius,\
                                    self.stepy, self.stepx, self.emailstr, self.hdf5path)
        self.dialog.setGroupTwoPara(self.bitDepth, self.dpf, self.rot90)
        self.dialog.setGroupThreePara(self.overlap, self.shareFrequency)
        self.dialog.setGroupFourPara(self.RMS, self.updateProbe, self.updateModes, self.phaseConstraint)

        self.dialog.flipScanAxisChangeSignal.connect(self.setflipScanAxis)
        self.dialog.mirror1stScanAxisChangeSignal.connect(self.setmirror1stScanAxis)
        self.dialog.mirror2ndScanAxisChangeSignal.connect(self.setmirror2ndScanAxis)
        self.dialog.jitterRadiusChangeSignal.connect(self.setjitterRadius)
        self.dialog.stepyChangeSignal.connect(self.setstepy)
        self.dialog.stepxChangeSignal.connect(self.setstepx)
        self.dialog.emailstrChangeSignal.connect(self.setemailstr)
        self.dialog.hdf5pathChangeSignal.connect(self.sethdf5path)

        self.dialog.bitDepthChangeSignal.connect(self.setbitDepth)
        self.dialog.dpfChangeSignal.connect(self.setdpf)
        self.dialog.rot90ChangeSignal.connect(self.setrot90)

        self.dialog.overlapChangeSignal.connect(self.setoverlap)
        self.dialog.shareFrequencyChangeSignal.connect(self.setshareFrequency)

        self.dialog.RMSChangeSignal.connect(self.setRMS)
        self.dialog.updateProbeChangeSignal.connect(self.setupdateProbe)
        self.dialog.updateModesChangeSignal.connect(self.setupdateModes)
        self.dialog.phaseConstraintChangeSignal.connect(self.setphaseConstraint)

        self.dialog.show()

    def setflipScanAxis(self, flipScanAxis):

        self.flipScanAxis=int(flipScanAxis)

    def setmirror1stScanAxis(self, mirror1stScanAxis):

        self.mirror1stScanAxis=int(mirror1stScanAxis)

    def setmirror2ndScanAxis(self, mirror2ndScanAxis):

        self.mirror2ndScanAxis=int(mirror2ndScanAxis)

    def setjitterRadius(self, jitterRadius):

        self.jitterRadius=int(jitterRadius)

    def setstepy(self, stepy):

        self.stepy=str(stepy)

    def setstepx(self, stepx):

        self.stepx=str(stepx)

    def setemailstr(self, emailstr):

        self.emailstr=str(emailstr)

    def sethdf5path(self, hdf5path):

        self.hdf5path=str(hdf5path)

    def setbitDepth(self, bitDepth):

        self.bitDepth=int(bitDepth)

    def setdpf(self, dpf):

        self.dpf=int(dpf)

    def setrot90(self, rot90):

        self.rot90=int(rot90)

    def setoverlap(self, overlap):

        self.overlap=int(overlap)

    def setshareFrequency(self, shareFrequency):

        self.shareFrequency=int(shareFrequency)

    def setRMS(self, RMS):

        self.RMS=int(RMS)

    def setupdateProbe(self, updateProbe):

        self.updateProbe=int(updateProbe)

    def setupdateModes(self, updateModes):

        self.updateModes=int(updateModes)

    def setphaseConstraint(self, phaseConstraint):

        self.phaseConstraint=int(phaseConstraint)

    def refreshData(self, cmdstrs, jobIDs):
        """
        Update the cmd string and job id
        :param cmdstrs:
        :param jobIDs:
        """
        self.jobIDs = jobIDs
        self.cmdstrs = cmdstrs
        # print self.cmdstrs, self.jobIDs

    def ePIEFile(self):
        """
        Browser the data folder path

        """
        dialog = QFileDialog()
        folder_path=dialog.getExistingDirectory(None, "Select Folder")

        self.ePIEdir=folder_path
        self.output_dir=os.path.join(self.ePIEdir,'data', self.prefix)
        self.create_dest_dirs(self.output_dir)
        self.ePIEdirlb.setText(str(folder_path))


    def browserMDA(self):
        """
        Browser the mda file

        """
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")

        formats = (["*.{0}".format(str("mda").lower())])
        fnames = QFileDialog.getOpenFileNames(self,
                                                    "MDA open -- Choose MDA", dir,
                                                    "MDA files ({0})".format(" ".join(formats)))[0]
		
        if fnames:
            #beamline=self.beamlineCbox.currentIndex()
            self.fplw.clear()
            self.fplw.addItems([fn.split('/mda/')[1] for fn in fnames])
            self.fplw.setCurrentRow(0)
            f = fnames[0].split('/mda/')
            self.fple.setText(f[0]+'/mda/')
            self.userdata_dir=str(f[0])

            BLmarker=str(f[1].split('_')[0])+'_'

            # clear the scanNo and filled with new mda scan number
            self.scanNo = []
            for fn in fnames:
                mdsStr = fn.split('/mda/')
                #self.fplw.addItem(mdsStr[1])
                #self.fplw.setCurrentRow(0)
                tempStr = mdsStr[1].split(BLmarker)
                scanStr = tempStr[1].split('.mda')
                self.scanNo.append(int(scanStr[0]))

                #if not os.path.exists(fp):
                    #QMessageBox.warning(self, 'Message', "Data file"+fp+" did not exist, please rechoose MDA files")
                    #return

            if self.userdata_dir:
                self.mda_path=os.path.join(self.userdata_dir,'mda/'+BLmarker+'{:04d}.mda')
                self.log_dir=os.path.join(self.userdata_dir,'log')
                self.logCreate()

    def resultDirectorySet(self):
        if self.userdata_dir:
            results_path=os.path.join(self.userdata_dir,'results')
            self.create_dest_dirs(results_path)
            os.system('rm data')
            os.system('ln -sf {:s} data'.format(results_path))
            self.log.info('data directory soft-linked to:'+results_path)
        else:
            self.log.info('Please select data directory!')

    def browserProbeGuess(self):
        """
        Browser the probeGuess file

        """
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")

        formats = (["*.{0}".format(str("csv").lower())])
        fname = QFileDialog.getOpenFileName(self,
                                                    "CSV open -- Choose CSV", dir,
                                                    "CSV files ({0})".format(" ".join(formats)))

        if fname[0]:
            self.probeGuess = fname[0]
            self.probeGuessle.setText(fname[0])

    def browserPositionFile(self):
        """
        Browser the position file

        """
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")

        formats = (["*.{0}".format(str("csv").lower())])
        fname = QFileDialog.getOpenFileName(self,
                                                    "CSV open -- Choose CSV", dir,
                                                    "CSV files ({0})".format(" ".join(formats)))

        if fname[0]:
            self.positionFile = fname[0]
            self.positionFilele.setText(fname[0])

    def objectGuessFile(self):
        """
        Browser the object guess file

        """
        dir = (os.path.dirname(self.filename)
               if self.filename is not None else ".")

        formats = (["*.{0}".format(str("csv").lower())])
        fname = QFileDialog.getOpenFileName(self,
                                                    "CSV open -- Choose CSV", dir,
                                                    "CSV files ({0})".format(" ".join(formats)))

        if fname[0]:
            self.objectGuessFile = fname[0]
            self.objectGuessle.setText(fname[0])



    def apply(self):
        """
        Get the parameter and run the ePIElib

        :return:
        """

        if(not self.fple.text()):
            QMessageBox.warning(self, 'Message', "Please choose MDA files")
            return

        if(self.distancele.text()):
            self.distance = float(self.distancele.text())
        if(self.pixelsizele.text()):
            self.pixelSize = int(self.pixelsizele.text())
        if(self.sizele.text()):
            self.diffSize = int(self.sizele.text())
        if(self.energyle.text()):
            self.energy = float(self.energyle.text())
        if(self.probele.text()):
            self.beamsize = float(self.probele.text())
        if(self.probemle.text()):
            self.pNum = int(self.probemle.text())
        if(self.tsle.text()):
            self.ts = int(self.tsle.text())
        if(self.fsle.text()):
            self.fs = int(self.fsle.text())
        if(self.nyle.text()):
            self.ny = int(self.nyle.text())
        if(self.itsle.text()):
            self.its = int(self.itsle.text())
        if(self.visitle.text()):
            self.visit = int(self.visitle.text())
            if(self.visit>=self.its):
                self.visit=self.its-1
        if(self.Tle.text()):
            self.T = int(self.Tle.text())
        if(self.algorithmle.text()):
            self.algorithm = str(self.algorithmle.text())

        self.stxm_dpc_only = int(self.stxmCheckBox.isChecked())
        self.det_mod = int(self.detmodCheckBox.isChecked())
        self.get_pos = int(self.getposCheckBox.isChecked())


        if( str(self.prele.text()) != self.prefix and self.prele.text()):
            self.prefix = str(self.prele.text())
            self.output_dir=os.path.join(self.ePIEdir,'data', self.prefix)
            self.create_dest_dirs(self.output_dir)

        os.chdir(self.ePIEdir) # the folder where the ePIE at

        cmdstrs = []
        jobIDs = []

        # start to read in the parameters from the MDA file
        for i in range(len(self.scanNo)):
            mdaCheck=self.mda_path.format(self.scanNo[i])
            h5Check=os.path.join(self.userdata_dir,'ptycho/scan{:03d}'.format(self.scanNo[i]))
            if not os.path.exists(mdaCheck):
                self.log.info("Can't find:"+mdaCheck)
            elif not os.path.exists(h5Check):
                self.log.info("Can't find:"+h5Check)

            else:
                mda_data=readMDA(mdaCheck)
                #self.log.info('Reading:'+mdaCheck)
                #Getting scan dimensions and diffraction pattern center
                STXM=np.array(mda_data[2].d[7].data)
                HDPC=np.array(mda_data[2].d[8].data) # p300K:Stats1:CentroidX_RBV
                VDPC=np.array(mda_data[2].d[9].data) # p300K:Stats1:CentroidY_RBV
                ny,nx=STXM.shape[0]-self.fs+1,STXM.shape[1]
                # qy,qx=VDPC.mean(),HDPC.mean()
                scanid=np.array(mda_data[2].d[18].data)
                scanid=int(scanid[1,1])
                
                qy,qx=256,256  #VDPC.mean(),HDPC.mean()

                if(self.nyle.text()):
                    ny = int(self.nyle.text())

                if(self.diffpattyle.text()):
                    qy = float(self.diffpattyle.text())
                if(self.diffpattxle.text()):
                    qx = float(self.diffpattxle.text())

                #Getting scan step size
                ypos=np.array(mda_data[1].p[0].data)
                xpos=np.array(mda_data[2].p[0].data)
               #stepy=(ypos.max()-ypos.min())/(ny-0)*1000 #nm #By Deng
                stepy=(ypos[ny-1]-ypos[0])/(ny-1)*1000 #nm

                stepx=(xpos[1,:].max()-xpos[1,:].min())/(nx-1+2)*1000 #nm

                #Getting X-ray energy
                # energy=10 #mda_data[1].d[0].data[0]

                # energy got from the mda file???
                if(self.energyle.text()):
                    energy = float(self.energyle.text())
                else:
                    energy = mda_data[1].d[0].data[0]

                self.energy = energy
                wavelength=self.energy_to_wavelength(energy)*1.e9 #nm

                #Calculating real space pixel size
                dx_s= wavelength*self.distance/(self.diffSize*self.pixelSize*1.e-6)  #nm

                #logging parameters
                # self.log.info('Scan grid dimensions: Y-{:d}, X-{:d}'.format(ny,nx))
                # self.log.info('Step size:Y-{:.1f} nm, X-{:.1f} nm'.format(stepy,stepx))
                # self.log.info('Diffraction pattern center: Y-{:.1f}, X-{:.1f}'.format(qy,qx))
                # self.log.info('X-ray Energy: {:.3f} keV'.format(energy))
                # self.log.info('Detector distance: {:.3f} m'.format(self.distance))
                # self.log.info('Reconstructed pixel size: {:.3f} nm'.format(dx_s))

                beamline=int(self.beamlineCbox.currentIndex())
 
                if beamline==0:
                    fp=os.path.join(h5Check,'scan{:03d}_#04d.h5'.format(self.scanNo[i])) #diffraction pattern path
                    diff_path=fp.split('#')[0]+'{:04d}.h5'
                elif beamline==1:
                    fp=os.path.join(h5Check,'scan{:03d}_#04d.h5'.format(self.scanNo[i])) #diffraction pattern path
                    diff_path=fp.split('#')[0]+'{:04d}.h5'
                else:
                    fp=os.path.join(h5Check,'scan{:03d}_{:d}_data_#06d.h5'.format(self.scanNo[i], scanid)) #diffraction pattern path
                    diff_path=fp.split('#')[0]+'{:06d}.h5'


                #logging parameters
                self.log.info('Scan grid dimensions: Y-{:d}, X-{:d}'.format(ny,nx))
                self.log.info('Step size:Y-{:.1f} nm, X-{:.1f} nm'.format(stepy,stepx))
                self.log.info('Diffraction pattern center: Y-{:.1f}, X-{:.1f}'.format(qy,qx))
                self.log.info('X-ray Energy: {:.3f} keV'.format(energy))
                self.log.info('Detector distance: {:.3f} m'.format(self.distance))
                self.log.info('Reconstructed pixel size: {:.3f} nm'.format(dx_s))

            if self.stxm_dpc_only==1:
                self.log.info('Calculating STXM_DPC: {:d}/{:d}'.format(i+1,len(self.scanNo)))
                dpc_path=os.path.join(self.userdata_dir,'STXM_DPC')
                self.create_dest_dirs(dpc_path)

                STXM_DPC=self.stxm_dpc(diff_path,nx, ny, self.fs)

                self.dpc_data_ready(STXM_DPC[1], STXM_DPC[2])
                # disImg = np.concatenate((STXM_DPC[1], STXM_DPC[2]), axis=0)
                # disImg = STXM_DPC[1]
                # img = scipy.misc.toimage(disImg)
                # self.display_image(img)


                np.save(dpc_path+'/'+'scan{:03d}'.format(self.scanNo[i])+'__aDPC.npy', STXM_DPC)
                #np.savetxt(dpc_path+'/'+'scan{:03d}'.format(self.scanNo[i])+'__HDPC.csv', STXM_DPC[2], delimiter=",")
                #self.imgPathLlbl.setText("Reconstruct: "+dpc_path+'/'+'scan{:03d}'.format(self.scanNo[i])+'__VDPC')
                #self.imgPathLlbr.setText("Reconstruct: "+dpc_path+'/'+'scan{:03d}'.format(self.scanNo[i])+'__HDPC')
                self.imgPathLlbl.setText('scan{:03d}'.format(self.scanNo[i])+'__VDPC')
                self.imgPathLlbr.setText('scan{:03d}'.format(self.scanNo[i])+'__HDPC')

                im=Image.fromarray(STXM_DPC[0])
                im.save(dpc_path+'/scan{:03d}'.format(self.scanNo[i])+'_STXM.tif')
                im=Image.fromarray(STXM_DPC[1])
                im.save(dpc_path+'/scan{:03d}'.format(self.scanNo[i])+'_VDPC.tif')
                im=Image.fromarray(STXM_DPC[2])
                im.save(dpc_path+'/scan{:03d}'.format(self.scanNo[i])+'_HDPC.tif')

                self.log.info('Saving STXM_DPC to:'+dpc_path)

            elif self.det_mod==1:
                self.log.info('Calculating det_mod.npz: {:d}/{:d}'.format(i+1,len(self.scanNo)))
                detmod_path=os.path.join(self.userdata_dir,'det_mod')
                self.create_dest_dirs(detmod_path)

                diffs=self.diff_stack(diff_path,nx,ny,int(qx),int(qy),self.fs)
                np.savez(detmod_path+'/scan{:03d}_det_mod.npz'.format(self.scanNo[i]), *[diffs[i] for i in range(len(diffs))])

                self.log.info('Saving det_mod.npz to:'+detmod_path)

            elif self.get_pos==1:
                self.log.info('Generating position files: {:d}/{:d}'.format(i+1,len(self.scanNo)))
                getpos_path=os.path.join(self.userdata_dir,'positions')
                self.create_dest_dirs(getpos_path)
                x_pos=np.array(mda_data[2].d[13].data).flatten()*1e-6
                y_pos=np.array(mda_data[2].d[14].data).flatten()*1e-6
                x_pos-=x_pos.mean()
                y_pos-=y_pos.mean()
                pos_stack=np.vstack((y_pos,x_pos)).T
                np.savetxt(getpos_path+'/scan{:03d}_pos.csv'.format(self.scanNo[i]),pos_stack,delimiter=',')

                self.log.info('Saving position file to:'+getpos_path)

            else:
                jobID='{:s}_{:.3f}keV_scan{:03d}_p{:d}_s{:d}_ts{:d}_i{:d}'.format(self.prefix,energy,
                        self.scanNo[i],self.pNum, self.diffSize,self.ts,self.its) #file saved name
                s=glob.glob(self.userdata_dir+'/positions/scan{:03d}_pos.csv'.format(self.scanNo[i]))

                if len(s)!=0:
                    self.positionFile=s[0]
                    self.log.info("Loading position file:"+s[0])
                else:
                    if not self.positionFilele.text():
                        self.positionFile=''
                        self.log.info("Can't find:"+self.userdata_dir+'/positions/scan{:03d}_pos.csv'.format(self.scanNo[i]))
                        self.log.info("Setting the default positions.")

                s1=glob.glob(self.userdata_dir+'/probes/scan{:03d}_probe.csv'.format(self.scanNo[i]))
                if len(s1)!=0:
                    self.probeGuess=s1[0]
                    self.log.info("Loading intial probe function:"+s1[0])
                else:
                    if not self.probeGuessle.text():
                        self.probeGuess=''
                        self.log.info("Can't find:"+self.userdata_dir+'/probes/scan{:03d}_probe.csv'.format(self.scanNo[i]))
                        self.log.info("Setting the default probe function.")

                s1=glob.glob(self.userdata_dir+'/objectguess/scan{:03d}_object.csv'.format(self.scanNo[i]))
                if len(s1)!=0:
                    self.objectGuess=s1[0]
                    self.log.info("Loading intial objecdt function:"+s1[0])
                else:
                    if not self.probeGuessle.text():
                        self.probeGuess=''
                        self.log.info("Can't find:"+self.userdata_dir+'/objectguess/scan{:03d}_object.csv'.format(self.scanNo[i]))
                        self.log.info("Setting the default object function.")

                self.log.info('************************************************************************')
                self.log.info('Start {:d}/{:d} reconstruction'.format(i+1,len(self.scanNo)))
                self.log.info('./ptycho -jobID={:s} -fp={:s} -fs={:d} -beamSize={:.1f}e-9 -qxy={:.1f},{:.1f} \
                          -scanDims={:d},{:d} -step={:.1f}e-9,{:.1f}e-9 -probeModes={:d} -i={:d} -rotate90={:d} \
                         -sqrtData -fftShiftData -threshold={:d} -size={:d} -lambda={:f}e-9 -dx_d={:d}e-6 \
                         -z={:.3f} -dpf={:d} -probeGuess={:s} -updateProbe={:d} -updateModes={:d} -lf={:s} -bitDepth={:d} -RMS={:d}\
                         -phaseConstraint={:d} -bitDepth={:d} -flipScanAxis={:d} -mirror1stScanAxis={:d} -mirror2ndScanAxis={:d} \
                         -jitterRadius={:d} -hdf5path={:s}'.format(jobID,fp,self.fs, self.beamsize,qy,qx,ny,nx,stepy,stepx, \
                                                      self.pNum,self.its, self.rot90, self.ts, self.diffSize, wavelength, \
                                                      self.pixelSize, self.distance, nx, self.probeGuess, self.updateProbe,\
                                                    self.updateModes, self.positionFile, self.bitDepth, self.RMS, \
                                                       self.phaseConstraint, self.bitDepth, self.flipScanAxis, self.mirror1stScanAxis, \
                                                        self.mirror2ndScanAxis, self.jitterRadius, self.hdf5path))
                self.log.info('************************************************************************')

                beginning = time.time()

                cmdstr= './ptycho -jobID={:s} -fp={:s} -fs={:d} -beamSize={:.1f}e-9 -qxy={:.1f},{:.1f} \
                          -scanDims={:d},{:d} -step={:.1f}e-9,{:.1f}e-9 -probeModes={:d} -i={:d} -rotate90={:d} \
                         -sqrtData -fftShiftData -threshold={:d} -size={:d} -lambda={:f}e-9 -dx_d={:d}e-6 \
                         -z={:.3f} -dpf={:d} -probeGuess={:s} -updateProbe={:d} -updateModes={:d} -lf={:s} -bitDepth={:d} -RMS={:d}\
                         -phaseConstraint={:d} -bitDepth={:d} -flipScanAxis={:d} -mirror1stScanAxis={:d} -mirror2ndScanAxis={:d} \
                         -jitterRadius={:d} -hdf5path={:s}'.format(jobID,fp,self.fs, self.beamsize,qy,qx,ny,nx,stepy,stepx, \
                                                      self.pNum,self.its, self.rot90, self.ts, self.diffSize, wavelength, \
                                                      self.pixelSize, self.distance, nx, self.probeGuess, self.updateProbe,\
                                                    self.updateModes, self.positionFile, self.bitDepth, self.RMS, \
                                                       self.phaseConstraint, self.bitDepth, self.flipScanAxis, self.mirror1stScanAxis, \
                                                        self.mirror2ndScanAxis, self.jitterRadius, self.hdf5path)

                # cmdstr= './ptycho -jobID={:s} -fp={:s} -fs={:d} -beamSize={:.1f}e-9 -qxy={:.1f},{:.1f} \
                #          -scanDims={:d},{:d} -step={:.1f}e-9,{:.1f}e-9 -probeModes={:d} -i={:d} -rotate90={:d} \
                #          -sqrtData -fftShiftData -threshold={:d} -size={:d} -lambda={:f}e-9 -dx_d={:d}e-6 \
                #          -z={:.3f} -dpf={:d} -probeGuess={:s} -updateProbe=20 -updateModes=30 -lf={:s}'.format(jobID,fp,self.fs, self.beamsize,qy,qx,ny,nx,stepy,stepx,
                #                                       self.pNum,self.its, self.rot90, self.ts, self.diffSize, wavelength,
                #                                       self.pixelSize, self.distance,nx, self.probeGuess, self.positionFile)


                self.jobID = jobID
                self.qxy = (int(qy), int(qx))
                self.scan_dims = (ny-self.fs+1, nx)
                self.step = (stepy*1e-9, stepx*1e-9)
                self.probeModes = int(self.pNum)
                self.dpf = int(nx)
                self.sqrt_data = 1
                self.fft_shift_data = 1


                cmdlist = {'fp':fp, 'fs': self.fs, 'size':self.diffSize, 'threshold':self.ts, \
                            'rotate90':self.rot90, 'sqrt_data':self.sqrt_data, \
                            'fft_shift_data':self.fft_shift_data, 'flipScanAxis':self.flipScanAxis, 'beamSize':self.beamsize*1e-9, \
                            'energy':self.energy,'dx_d':self.pixelSize*1e-6,'z':self.distance, 'simulate': self.simulate, \
                            'scan_dims': self.scan_dims, 'step':self.step, \
                            'probeGuess': self.probeGuess, 'objectArray':self.objectArray, 'i':self.its, 'jobID': self.jobID, 'qxy':self.qxy, \
                            'probeModes':self.probeModes, 'jitterRadius':self.jitterRadius, 'shareFrequency':self.shareFrequency, 'dpf':self.dpf }


                self.cmdlists.append(cmdlist)

                # now = time.time()
                # log.info('{:5.2f} seconds have elapsed in {:d} iterations [{:2.2f} sec/it/mode] with {:d} probe modes'
                #          .format(now-beginning, self.its, (now-beginning)/self.its/self.pNum, self.pNum))


                jobIDs.append(jobID)
                cmdstrs.append(cmdstr)

                self.cmdstrs = cmdstrs
                self.jobIDs = jobIDs

        # Start ePIE
        if self.stxm_dpc_only==0 and self.det_mod==0 and self.get_pos==0:
            self.runEPIE()

    def loadSetting(self):
        """
        Load the setting when you first start the GUI

        """
        self.settings = QSettings('UChicago', 'ptychoRecon')

        self.beamline = self.settings.value('self_beamline', type = int)
        self.beamlineCbox.setCurrentIndex(self.beamline)

        # self.stxm_dpc_only = self.settings.value('self_stxm_dpc_only', type = int)
        # self.stxmCheckBox.setChecked(self.stxm_dpc_only)
        #
        # self.det_mod = self.settings.value('self_det_mod', type = int)
        # self.detmodCheckBox.setChecked(self.det_mod)

        self.energy = self.settings.value('self_energy', type = float)
        self.energyle.setText(str(self.energy))

        self.beamsize = self.settings.value('self_beamsize', type = float)
        self.probele.setText(str(self.beamsize))

        self.prefix = self.settings.value('self_prefix', type = str)
        self.prele.setText(self.prefix)

        self.pNum = self.settings.value('self_pNum', type = int)
        self.probemle.setText(str(self.pNum))

        self.gpuNo = self.settings.value('self_gpuNo', type = int)
        self.gpuCbox.setCurrentIndex(self.gpuNo)

        self.rot90 = self.settings.value('self_rot90', type = int)
        self.rotateCbox.setCurrentIndex(self.rot90)

        self.diffSize = self.settings.value('self_diffSize', type = int)
        self.sizele.setText(str(self.diffSize))

        self.pixelSize = self.settings.value('self_pixelSize', type = int)
        self.pixelsizele.setText(str(self.pixelSize))

        self.distance = self.settings.value('self_distance', type = float)
        self.distancele.setText(str(self.distance))

        self.ts = self.settings.value('self_ts', type = int)
        self.tsle.setText(str(self.ts))

        self.fs = self.settings.value('self_fs', type = int)
        self.fsle.setText(str(self.fs))

        self.its = self.settings.value('self_its', type = int)
        self.itsle.setText(str(self.its))

        self.visit = self.settings.value('self_visit', type = int)
        self.visitle.setText(str(self.visit))

        # self.emailstr = self.settings.value('self_emailstr', type = str)
        # self.emaille.setText(self.emailstr)

    def logCreate(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            open(self.log_dir+'/Ptycholib.log', 'w')
            os.close(self.log_dir+'/Ptycholib.log')
            os.chown(self.log_dir+'/Ptycholib.log', os.getlogin())

        self.log = logging.getLogger('Ptycholib')
        self.log.setLevel(logging.DEBUG)
        # Create a file handler for debug level and above
        fh = logging.handlers.RotatingFileHandler(self.log_dir+'/Ptycholib.log', maxBytes = 1e6, backupCount=5)
        fh.setLevel(logging.DEBUG)
        # Create console handler with higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        ch.setFormatter(console_formatter)
        fh.setFormatter(file_formatter)
        self.log.addHandler(ch)
        self.log.addHandler(fh)

    def create_dest_dirs(self, dirname):

        if not os.path.exists(dirname):
            os.makedirs(dirname, mode=0o777)
            print('Creating directory:'+dirname)
        else:
            print(dirname + " already exists!")

    def energy_to_wavelength(self, energy):
        """
        Converts an energy in keV to wavelength in metres.
        """
        wl = spc.physical_constants['Planck constant'][0]
        wl *= spc.speed_of_light
        wl /= (energy*1e3*spc.elementary_charge)
        return wl

    def sliderval(self):
        self.loadFile(self.fnames[self.s1.value()])

    def stxm_dpc(self, file_path, nx, ny,fs=1):
        l=[]
        com=[]
        for i in range(fs,fs+ny):
            data=self.read_h5(file_path.format(i))
            print('Processing Line {:d}/{:d}'.format(i-fs+1,ny))
            for j in range(nx):
                l.append(np.sum(data[j]))
                com.append(spn.center_of_mass(data[j]))
        stxm=np.array(l,float).reshape([ny,nx])
        dpc=np.array(com).reshape((ny,nx,2))
        stxm_dpc=[]
        stxm_dpc.append(stxm)
        stxm_dpc.append(dpc[:,:,0])
        stxm_dpc.append(dpc[:,:,1])
        return np.array(stxm_dpc,dtype=float)

    def diff_stack(self, file_path, nx, ny,qx,qy,fs=1):
        diffs=[]
        for i in range(fs,fs+ny):
            data=self.read_h5(file_path.format(i))
            print('Processing Line {:d}/{:d}'.format(i-fs+1,ny))
            for j in range(nx):
                diffs.append(np.fft.fftshift(data[j][qy-int(self.diffSize/2):qy+int(self.diffSize/2),qx-int(self.diffSize/2):qx+int(self.diffSize/2)]))
        return diffs

    def read_h5(self, filename):
        h5_file_path='entry/instrument/detector/data'
        h5_open=h5py.File(filename,'r')
        data=h5_open[h5_file_path].value
        h5_open.close()
        return data

    def loadFile(self, fname=None):
        # if fname.find("__aDPC"):
        if '__aDPC' in fname:
            fnamel = fname.split('__')[0]+'__VDPC'
            fnamer = fname.split('__')[0]+'__HDPC'
            self.imgPathLlbl.setText(fnamel)
            self.imgPathLlbr.setText(fnamer)
            ob=np.load(str(fname))
            #obl = np.genfromtxt(unicode(fnamel), delimiter=",", dtype=None)
            #obr = np.genfromtxt(unicode(fnamer), delimiter=",", dtype=None)
            self.dpc_data_ready(ob[1], ob[2])
        else:
            ob = self.opencsv(fname)
            self.cmd_data_ready(ob, fname)

    def opencsv(self, stringname):
        ob=np.genfromtxt(str(stringname),delimiter=',',dtype=complex)
        return ob

if __name__ == "__main__":

    app = QApplication(sys.argv)
    widget = ImgWidget()
    widget.loadSetting()
    widget.resize(1600, 800)
    widget.show()

    sys.exit(app.exec_())
