# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:18:45 2023

@author: YOGB
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
import pandas as pd

class CPT_LIQ:
    def __init__(self, file, Mw, PGA, gwd):
        self.file = file
        self.Mw = Mw
        self.PGA = PGA
        self.gwd = gwd

    def LDCR(self):
        wb = self.file
        sheet_general = wb['general']
        sheet_cpt = wb['cpt']
        
        CPT_label = sheet_general.cell(2,2).value
        Z = sheet_general.cell(5,2).value
        gwl = Z - self.gwd
        # self.gwd = Z - gwl
        elev_unit = sheet_general.cell(7,2).value
        
        z, qc, fs, elev = [],[],[],[]
        for i in range(sheet_cpt.max_row - 1):
            z.append(sheet_cpt.cell(i+2,1).value)
            qc.append(np.abs(sheet_cpt.cell(i+2,2).value))
            fs.append(np.abs(sheet_cpt.cell(i+2,3).value))
            elev.append(Z - z[i])                                 
        
        ndata = len(z)
         
        rf = [(i/j*100) for i,j in zip(fs,qc)]
        
        # unit weight (gamma) =========================================================
        gamma = [10*(0.27*np.log10(x)+0.36*np.log10(y/0.1)+1.236) for x,y in zip(rf,qc)]
        
        # vertical stress (sigma) =====================================================
        sig_tot = np.zeros(ndata)
        u0 = np.zeros(ndata)
        dz = z[1]-z[0]
        sig_tot[0] = z[0]*gamma[0]
        for i in range(ndata-1):
            sig_tot[i+1] = (sig_tot[i] + dz*gamma[i+1])
            if z[i+1] > self.gwd:
                u0[i+1] = (u0[i] + dz*10.0)
        sig_eff = sig_tot - u0
        
        # I-SBT (Robertson) ===========================================================
        Qt = [(x*1000-y)/z for x,y,z in zip(qc,sig_tot,sig_eff)]
        Fr = [a*1000/(b*1000-c)*100 for a,b,c in zip(fs,qc,sig_tot)]
        Ic = [np.sqrt((3.47-np.log10(x))**2 + (np.log10(y)+1.22)**2) for x,y in zip(Qt,Fr)]
        
        index = []                  # this is only for plotting purpose
        for i in range(ndata):
            if Ic[i] > 3.6:
                index.append(2)
            elif Ic[i] > 2.95 and Ic[i] <=3.6:
                index.append(3)
            elif Ic[i] > 2.60 and Ic[i] <=2.95:
                index.append(4)
            elif Ic[i] > 2.05 and Ic[i] <=2.60:
                index.append(5)
            elif Ic[i] > 1.31 and Ic[i] <=2.05:
                index.append(6)
            else:
                index.append(7)
        
        # IB (Robertson, 2016) ========================================================
        n = [0.381*a + 0.05*b/100 - 0.15 for a,b in zip(Ic,sig_eff)]
        n2 = []
        for x in n:
            if x > 1:
                n2.append(1)
            else:
                n2.append(x)
        
        Qtn = [((a*1000-b)/100)*(100/c)**d for a,b,c,d in zip(qc,sig_tot,sig_eff,n)]
        
        IB = [100*(a+10)/(a*b+70) for a,b in zip(Qtn,Fr)]
        CD = [(a-11)*(1+0.06*b)**17 for a,b in zip(Qtn,Fr)]
        
        label_CD, color_CD = [],[]
        for i in range(ndata):
            if IB[i] > 32 and CD[i] > 70:
                label_CD.append("SD")
                color_CD.append("tomato")
            elif IB[i] <= 32 and IB[i] > 22 and CD[i] > 70:
                label_CD.append("TD")
                color_CD.append("yellowgreen")
            elif IB[i] <= 22 and CD[i] > 70:
                label_CD.append("CD")
                color_CD.append("deepskyblue")
            elif IB[i] > 32 and CD[i] <= 70:
                label_CD.append("SC")
                color_CD.append("salmon")
            elif IB[i] <= 32 and IB[i] > 22 and CD[i] <= 70:
                label_CD.append("TC")
                color_CD.append("greenyellow")
            elif IB[i] <= 22 and CD[i] <= 70:
                label_CD.append("CC")
                color_CD.append("lightskyblue")
        
        # FILTERING FOR ZA, ZB ========================================================
        bot_pot = Z - 15
        up_pot = gwl
        
        filter1 = []
        for i in range(ndata):
            if elev[i] > up_pot:
                filter1.append(0)
            elif elev[i] < up_pot and elev[i] > bot_pot and Ic[i] <= 2.6:
                filter1.append(1)
            elif elev[i] < up_pot and elev[i] > bot_pot and Ic[i] > 2.6:
                filter1.append(2)
            elif elev[i] < bot_pot:
                filter1.append(0)
        
        nwindow = int(0.25/dz)
        
        i = 0
        filter1_ma = []
        while i < len(filter1)-nwindow+1:
            window_avg = np.sum(filter1[i:i+nwindow])/nwindow
            filter1_ma.append(window_avg)
            i+=1
        
        elev2 = elev[:-nwindow+1]
        
        wadahza = []
        for j in range(len(filter1_ma)):
            if filter1_ma[j] == 1:
                wadahza.append(j)
        indexza = np.min(wadahza)
        ZA = elev2[indexza]
        
        wadahzb = []
        for k in range(len(filter1_ma)):
            if elev2[k] < ZA and filter1_ma[k] == 2:
                wadahzb.append(k)
        indexzb = np.min(wadahzb)
        ZB = elev2[indexzb]
        
        # return(ZB)
        
        # LIQUEFACTION TRIGGERING PROCEDURE ===========================================
        alpha = [1-0.4113*a**0.5+0.04052*a+0.001753*a**1.5 for a in z]
        beta = [1-0.4177*a**0.5+0.05729*a-0.006205*a**1.5+0.00121*a**2 for a in z]
        rd = [a/b for a,b in zip(alpha,beta)]
        
        delta = np.zeros(len(z))
        for i in range(len(z)-1):
            delta[i+1] = np.abs((Ic[i+1]-Ic[i])/(z[i+1]-z[i]))
        
        check = []
        for i in range(len(delta)):
            if Ic[i] > 1.3 and Ic[i] < 3.6 and delta[i] > 2:
                check.append(1)
            else:
                check.append(-1)
        
        CN2 = (100/sig_eff)**0.5
        CN = []
        for j in range(ndata):
            if CN2[j] > 1.7:
                CN.append(1.7)
            else:
                CN.append(CN2[j])
        
        qc1n = [a*b*1000/100 for a,b in zip(CN,qc)]
        
        Kc = []
        for i in range(ndata):
            if Ic[i] <= 1.64:
                Kc.append(1.0)
            else:
                Kc.append(-0.403*Ic[i]**4+5.581*Ic[i]**3-21.63*Ic[i]**2+33.75*Ic[i]-17.88)
        
        qc1ncs = [a*b for a,b in zip(Kc,qc1n)]
        
        # K-SIGMA ---------------------------------------------------------------------
        
        f = 0.7
        Ksigma = [(a/100)**(f-1) for a in sig_eff]
        
        # MAGNITUDE SCALING FACTOR ----------------------------------------------------
        MSF = (self.Mw/7.5)**-2.56
        
        # CYCLIC RESISTANCE RATIO -----------------------------------------------------
        CRR_m75 = []
        for i in range(ndata):
            if qc1ncs[i] < 50:
                CRR_m75.append(0.833*(qc1ncs[i]/1000)+0.05)
            elif qc1ncs[i] >= 50 and qc1ncs[i] < 160:
                CRR_m75.append(93*(qc1ncs[i]/1000)**3+0.08)
            else:
                CRR_m75.append(1.0)
        
        CRR_m = [a*b*MSF for a,b in zip(CRR_m75,Ksigma)]
        
        # CYCLIC STRESS RATIO ---------------------------------------------------------
        CSR_m = [0.65*a/b*self.PGA*c*1.3 for a,b,c in zip(sig_tot,sig_eff,rd)]
        
        # FACTOR OF SAFETY ------------------------------------------------------------
        FS = []
        for i in range(ndata):
            if elev[i] > gwl:
                FS.append(5)
            elif elev[i] < gwl and Ic[i] > 2.6:
                FS.append(5)
            else:
                FS.append(CRR_m[i]/CSR_m[i])
        
        
        # PORE PRESSURE RATIO ---------------------------------------------------------
        ru = []
        for i in range(ndata):
            if FS[i] < 1:
                ru.append(1.0)
            elif FS[i] >= 1:
                ru.append(0.5+(math.asin(2*FS[i]**-5 - 1)/math.pi))
                
        # EXCESS HEAD -----------------------------------------------------------------
        h_exc = [a*b/9.81 for a,b in zip(ru,sig_eff)]
        h_eff = [a/9.81 for a in sig_eff]
        
        # VERTICAL PERMEABILITY -------------------------------------------------------
        kv = [10**(0.952-3.04*a) for a in Ic]
        kv_ratio = [a/(3*10**-5) for a in kv]
        
        # EJECTA DEMAND CALCULATION ---------------------------------------------------
        fungsi_LD = []
        for i in range(ndata):
            if elev[i] < ZA and elev[i] > ZB and h_exc[i] > z[i]:
                fungsi_LD.append(kv_ratio[i]*(h_exc[i]-z[i])*9.81)
            else:
                fungsi_LD.append(0)
        
        INT_LD = np.zeros(ndata)
        for i in range(ndata-1):
            INT_LD[i+1] = ((fungsi_LD[i+1]+fungsi_LD[i])/2*dz) + INT_LD[i]
        
        LD = np.max(INT_LD)
        
        # CRUST RESISTANCE CALCULATION ------------------------------------------------
        fungsi_CR = []
        for i in range(ndata):
            if elev[i] > ZA:
                if IB[i] > 22:
                    fungsi_CR.append(0.5*sig_eff[i]*math.tan(math.radians(33)))
                elif IB[i] <= 22:
                    fungsi_CR.append((qc[i]*1000-sig_tot[i])/15)
            elif elev[i] <= ZA:
                fungsi_CR.append(0)
        
        INT_CR = np.zeros(ndata)
        for i in range(ndata-1):
            INT_CR[i+1] = ((fungsi_CR[i+1]+fungsi_CR[i])/2*dz) + INT_CR[i]
        
        CR = np.max(INT_CR)
        
        # SEVERITY CLASSIFICATION -----------------------------------------------------
        coord1 = [[0,100,250],[2.5,2.5,25]]
        coord2 = [[0,90,250],[6,6,70]]
        coord3 = [[0,85,250],[15,15,150]]
        coord4 = [[0,75,250],[85,85,315]]
        
        line1 = interp1d(coord1[0], coord1[1])
        line2 = interp1d(coord2[0], coord2[1])
        line3 = interp1d(coord3[0], coord3[1])
        line4 = interp1d(coord4[0], coord4[1])      
        
        dot1 = line1(CR)
        dot2 = line2(CR)
        dot3 = line3(CR)
        dot4 = line4(CR)
        
        if LD < dot1:
            category = "None"
        elif LD >= dot1 and LD < dot2:
            category = "Minor"
        elif LD >= dot2 and LD < dot3:
            category = "Moderate"  
        elif LD >= dot3 and LD < dot4:
            category = "Severe"  
        elif LD >= dot4:
            category = "Extreme"  
        
        # SETTLEMENT CALCULATION ------------------------------------------------------
        epsv = np.zeros(ndata)
        for i in range(ndata):
            if FS[i] <= 0.5 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.5 and FS[i] <= 0.6 and qc1ncs[i] >= 33 and qc1ncs[i] <= 147 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.5 and FS[i] <= 0.6 and qc1ncs[i] > 147 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 2411*qc1ncs[i]**-1.45
            elif FS[i] > 0.6 and FS[i] <= 0.7 and qc1ncs[i] >= 33 and qc1ncs[i] <= 110 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.6 and FS[i] <= 0.7 and qc1ncs[i] > 110 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1701*qc1ncs[i]**-1.42
            elif FS[i] > 0.7 and FS[i] <= 0.8 and qc1ncs[i] >= 33 and qc1ncs[i] <= 80 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.7 and FS[i] <= 0.8 and qc1ncs[i] > 80 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1690*qc1ncs[i]**-1.46
            elif FS[i] > 0.8 and FS[i] <= 0.9 and qc1ncs[i] >= 33 and qc1ncs[i] <= 60 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.8 and FS[i] <= 0.9 and qc1ncs[i] > 60 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1430*qc1ncs[i]**-1.48
            elif FS[i] > 0.9 and FS[i] <= 1.0 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 64*qc1ncs[i]**-0.93
            elif FS[i] > 1.0 and FS[i] <= 1.1 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 11*qc1ncs[i]**-0.65
            elif FS[i] > 1.1 and FS[i] <= 1.2 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 9.7*qc1ncs[i]**-0.69  
            elif FS[i] > 1.2 and FS[i] <= 1.3 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 7.6*qc1ncs[i]**-0.71
            else:
                epsv[i] = 0.0
                
        sett = [a/100*dz for a in epsv]
        sett2 = np.cumsum(sett)
        sett3 = [np.max(sett2)-i for i in sett2]
        
        # CREATE DATAFRAME ============================================================
        tabel = {"depth_m":z, "elev":elev, "qc_MPa":qc, "fs_MPa": fs, "Rf_%":rf, "gamma_kN/m3":gamma,"sigtot_kPa":sig_tot, 
                  "u_kPa":u0, "sigeff":sig_eff, "Qt":Qt, "Fr":Fr, "Ic":Ic, "delta":delta, "check":check, "IB":IB, "CD":CD,
                  "rd":rd, "CN":CN, "qc1n":qc1n, "Kc":Kc, "qc1ncs":qc1ncs, "Ksigma":Ksigma, "CRR_m75":CRR_m75, "CRR_m":CRR_m,
                  "CSR_m":CSR_m, "FS":FS, 
                  "ru":ru, "h_exc_m":h_exc, "kv/kcs":kv_ratio, "LD":INT_LD, "CR":INT_CR, "epsv":epsv, "sett_m":sett3}
        
        df = pd.DataFrame(tabel)
        
        # PLOTTING ====================================================================
        ymin = np.min(elev)
        ymax = np.max(elev)
        
        fig, ax = plt.subplots(1, 5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(15,6))
        plt.suptitle('\n'+CPT_label+', LD = %.2f, CR = %.2f, Category = %s'%(LD,CR,category), size=12, x=0.1, y=1.03, horizontalalignment='left')
        ax1 = ax[0]
        ax1.plot(qc,elev,color='black', label="qc")
        ax1.plot([100,100],[0,10],color='green', label="Rf") # this is only dummy
        ax1.plot([0,15],[gwl,gwl],color='blue', linestyle="--", label="gwl", zorder=3)
        ax1.plot([0,15],[ZA,ZA],color="blue",linestyle="--",label="ZA")
        ax1.plot([0,15],[ZB,ZB],color="black",linestyle="--",label="ZB")
        ax1.set_ylim(ymin,ymax)
        ax1.set_xlim(0,15.0)
        # ax1.invert_yaxis()
        ax1.minorticks_on()
        ax1.grid(False, which='both')
        ax1.legend(loc=3)
        ax1.set_ylabel("Elev (%s)"%(elev_unit),size=12)
        ax1.set_xlabel("qc (MPa)",size=12)
        ax2 = ax1.twiny()
        ax2.plot(rf,elev,color='green', label="Rf")
        ax2.set_ylim(ymin,ymax)
        ax2.set_xlim(0,30.0)
        ax2.invert_xaxis()
        # ax2.invert_yaxis()
        ax2.set_xlabel("Rf (%)",size=12)
        for i in range(ndata-1):
            ax1.add_patch(patches.Rectangle((0,elev[i]),30,dz,facecolor=color_CD[i]))
        
        ax3 = ax[1]
        ax3.plot(Ic,elev,color='black',linewidth=1,zorder=3)
        ax3.set_ylim(ymin,ymax)
        ax3.set_xlim(1.0,4.0)
        # ax3.invert_yaxis()
        ax3.add_patch(patches.Rectangle((0,ymin),1.31,ymax-ymin,facecolor='goldenrod'))
        ax3.add_patch(patches.Rectangle((1.31,ymin),0.74,ymax-ymin,facecolor='khaki'))
        ax3.add_patch(patches.Rectangle((2.05,ymin),0.55,ymax-ymin,facecolor='lightsteelblue'))
        ax3.add_patch(patches.Rectangle((2.60,ymin),0.35,ymax-ymin,facecolor='yellowgreen'))
        ax3.add_patch(patches.Rectangle((2.95,ymin),0.65,ymax-ymin,facecolor='olivedrab'))
        ax3.add_patch(patches.Rectangle((3.6,ymin),0.4,ymax-ymin,facecolor='sienna'))
        ax3.yaxis.grid(which="minor")
        ax3.yaxis.grid(which="major")
        ax3.xaxis.grid(which="major")
        ax3.minorticks_on()
        ax3.text(1.05, ymin+1, "gravel - dense sand", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(1.55, ymin+1, "clean sand - silty sand", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(2.22, ymin+1, "silty sand - sandy silt", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(2.65, ymin+1, "clayey silt - silty clay", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(3.2, ymin+1, "silty clay - clay", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(3.65, ymin+1, "peat", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.set_xlabel("Ic",size=12)
        ax3.set_yticks([])
        
        ax4 = ax[2]
        ax4.plot(FS,elev,color="black")
        ax4.plot([1,1],[ymin,ymax],linestyle="--",color="black")
        for i in range(ndata-1):
            if FS[i+1] <= 1.0:
                ax4.fill_betweenx([elev[i],elev[i+1]],[FS[i],FS[i+1]],[1,1],color='salmon')
        ax4.set_ylim(ymin,ymax)
        ax4.set_xlim(0,2)
        # ax4.invert_yaxis()
        ax4.yaxis.grid(which="major")
        ax4.yaxis.grid(which="minor")
        ax4.xaxis.grid(which="major")
        ax4.grid(True)
        # ax4.set_xticks([])
        ax4.set_xlabel("FS",size=12)
        ax4.set_yticks([])
        # ax4.legend(bbox_to_anchor=(1,1), loc="upper left",edgecolor='none')
        
        ax5 = ax[3]
        ax5.plot(h_exc,elev,color="black")
        ax5.plot([0,20],[Z,Z-20],linestyle="--",color="black")
        ax5.plot(h_eff,elev,color="red",linestyle="-")
        for i in range(ndata-1):
            if elev[i+1] < ZA and elev[i+1] > ZB and h_exc[i+1] > z[i+1]:
                ax5.fill_betweenx([elev[i],elev[i+1]],[z[i],z[i+1]],[h_exc[i],h_exc[i+1]],color='salmon')
        ax5.set_ylim(ymin,ymax)
        ax5.set_xlim(0,20)
        # ax4.invert_yaxis()
        ax5.yaxis.grid(which="major")
        ax5.yaxis.grid(which="minor")
        ax5.xaxis.grid(which="major")
        ax5.grid(True)
        # ax4.set_xticks([])
        ax5.set_yticks([])
        ax5.set_xlabel("excess head (m)",size=12)
        
        
        ax6 = ax[4]
        ax6.plot(INT_LD,elev,color='black',label='LD')
        ax6.plot([dot1,dot1],[ymin,ymax],color='silver',linestyle='--')
        ax6.plot([dot2,dot2],[ymin,ymax],color='silver',linestyle='--')
        ax6.plot([dot3,dot3],[ymin,ymax],color='silver',linestyle='--')
        ax5.plot([dot4,dot4],[ymin,ymax],color='silver',linestyle='--')
        ax6.text(LD-0.1*LD, ymin+1, "category = %s"%(category), va='bottom', rotation=90, size=10, color="black",zorder=1)
        ax6.set_ylim(ymin,ymax)
        ax6.set_xlabel("LD (kN/m)",size=12)
        ax6.set_yticks([])
        ax6.set_xlim(0,LD*1.1)
        
        # ax6 = ax[4]
        # ax6.plot(sett3,elev,color='black')
        # ax6.text(np.max(sett3)-0.02, ymax-(0.02*(ymax-ymin)), "max settlement = %.2f mm"%(np.max(sett3)*1000), va='top', rotation=90, size=10, color="black",zorder=1)
        # ax6.set_ylim(ymin,ymax)
        # ax6.set_xlabel("induced settlement (m)",size=12)
        # ax6.xaxis.grid(which="major")
        # ax6.set_yticks([])
        # plt.tight_layout()
        
        return(fig,df,LD,CR,category)
    
    def LPI(self):
        wb = self.file
        sheet_general = wb['general']
        sheet_cpt = wb['cpt']
        
        CPT_label = sheet_general.cell(2,2).value
        Z = sheet_general.cell(5,2).value
        gwl = Z - self.gwd           
        # self.gwd = Z - gwl
        elev_unit = sheet_general.cell(7,2).value
        
        z, qc, fs, elev = [],[],[],[]
        for i in range(sheet_cpt.max_row - 1):
            z.append(sheet_cpt.cell(i+2,1).value)
            qc.append(np.abs(sheet_cpt.cell(i+2,2).value))
            fs.append(np.abs(sheet_cpt.cell(i+2,3).value))
            elev.append(Z - z[i])                                 
        
        ndata = len(z)
         
        rf = [(i/j*100) for i,j in zip(fs,qc)]
        
        # unit weight (gamma) =========================================================
        gamma = [10*(0.27*np.log10(x)+0.36*np.log10(y/0.1)+1.236) for x,y in zip(rf,qc)]
        
        # vertical stress (sigma) =====================================================
        sig_tot = np.zeros(ndata)
        u0 = np.zeros(ndata)
        dz = z[1]-z[0]
        sig_tot[0] = z[0]*gamma[0]
        for i in range(ndata-1):
            sig_tot[i+1] = (sig_tot[i] + dz*gamma[i+1])
            if z[i+1] > self.gwd:
                u0[i+1] = (u0[i] + dz*10.0)
        sig_eff = sig_tot - u0
        
        # I-SBT (Robertson) ===========================================================
        Qt = [(x*1000-y)/z for x,y,z in zip(qc,sig_tot,sig_eff)]
        Fr = [a*1000/(b*1000-c)*100 for a,b,c in zip(fs,qc,sig_tot)]
        Ic = [np.sqrt((3.47-np.log10(x))**2 + (np.log10(y)+1.22)**2) for x,y in zip(Qt,Fr)]
        
        index = []                  # this is only for plotting purpose
        for i in range(ndata):
            if Ic[i] > 3.6:
                index.append(2)
            elif Ic[i] > 2.95 and Ic[i] <=3.6:
                index.append(3)
            elif Ic[i] > 2.60 and Ic[i] <=2.95:
                index.append(4)
            elif Ic[i] > 2.05 and Ic[i] <=2.60:
                index.append(5)
            elif Ic[i] > 1.31 and Ic[i] <=2.05:
                index.append(6)
            else:
                index.append(7)
        
        # LIQUEFACTION TRIGGERING PROCEDURE ===========================================
        alpha = [1-0.4113*a**0.5+0.04052*a+0.001753*a**1.5 for a in z]
        beta = [1-0.4177*a**0.5+0.05729*a-0.006205*a**1.5+0.00121*a**2 for a in z]
        rd = [a/b for a,b in zip(alpha,beta)]
        
        delta = np.zeros(len(z))
        for i in range(len(z)-1):
            delta[i+1] = np.abs((Ic[i+1]-Ic[i])/(z[i+1]-z[i]))
        
        check = []
        for i in range(len(delta)):
            if Ic[i] > 1.3 and Ic[i] < 3.6 and delta[i] > 2:
                check.append(1)
            else:
                check.append(-1)
        
        CN2 = (100/sig_eff)**0.5
        CN = []
        for j in range(ndata):
            if CN2[j] > 1.7:
                CN.append(1.7)
            else:
                CN.append(CN2[j])
        
        qc1n = [a*b*1000/100 for a,b in zip(CN,qc)]
        
        Kc = []
        for i in range(ndata):
            if Ic[i] <= 1.64:
                Kc.append(1.0)
            else:
                Kc.append(-0.403*Ic[i]**4+5.581*Ic[i]**3-21.63*Ic[i]**2+33.75*Ic[i]-17.88)
        
        qc1ncs = [a*b for a,b in zip(Kc,qc1n)]
        
        # K-SIGMA ---------------------------------------------------------------------
        
        f = 0.7
        Ksigma = [(a/100)**(f-1) for a in sig_eff]
        
        # MAGNITUDE SCALING FACTOR ----------------------------------------------------
        MSF = (self.Mw/7.5)**-2.56
        
        # CYCLIC RESISTANCE RATIO -----------------------------------------------------
        CRR_m75 = []
        for i in range(ndata):
            if qc1ncs[i] < 50:
                CRR_m75.append(0.833*(qc1ncs[i]/1000)+0.05)
            elif qc1ncs[i] >= 50 and qc1ncs[i] < 160:
                CRR_m75.append(93*(qc1ncs[i]/1000)**3+0.08)
            else:
                CRR_m75.append(1.0)
        
        CRR_m = [a*b*MSF for a,b in zip(CRR_m75,Ksigma)]
        
        # CYCLIC STRESS RATIO ---------------------------------------------------------
        CSR_m = [0.65*a/b*self.PGA*c*1.3 for a,b,c in zip(sig_tot,sig_eff,rd)]
        
        # FACTOR OF SAFETY ------------------------------------------------------------
        FS = []
        for i in range(ndata):
            if elev[i] > gwl:
                FS.append(5)
            elif elev[i] < gwl and Ic[i] > 2.6:
                FS.append(5)
            else:
                FS.append(CRR_m[i]/CSR_m[i])
                
        # LPI PART ====================================================================
        F = []
        for i in range(len(FS)):
            if Ic[i]<2.6 and FS[i]<1.0 and z[i]>self.gwd and check[i]<0.0:
                F.append(1.0-FS[i])
            else:
                F.append(0.0)
        
        wz = [10.0 - 0.5*i for i in z]
        F_wz = [i*j for i,j in zip(F,wz)]
        
        INT = np.zeros(len(z))
        for i in range(len(z)-1):
            INT[i+1] = ((F_wz[i+1]+F_wz[i])/2*dz) + INT[i]
        
        LPI = np.max(INT)
        
        # SEVERITY CLASSIFICATION -----------------------------------------------------
        if LPI < 2:
            category = "Very low"
        elif LPI >= 2 and LPI < 5:
            category = "Low"
        elif LPI >= 5 and LPI < 15:
            category = "High"  
        elif LPI >= 15:
            category = "Very High"  
        
        # SETTLEMENT CALCULATION ------------------------------------------------------
        epsv = np.zeros(ndata)
        for i in range(ndata):
            if FS[i] <= 0.5 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.5 and FS[i] <= 0.6 and qc1ncs[i] >= 33 and qc1ncs[i] <= 147 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.5 and FS[i] <= 0.6 and qc1ncs[i] > 147 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 2411*qc1ncs[i]**-1.45
            elif FS[i] > 0.6 and FS[i] <= 0.7 and qc1ncs[i] >= 33 and qc1ncs[i] <= 110 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.6 and FS[i] <= 0.7 and qc1ncs[i] > 110 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1701*qc1ncs[i]**-1.42
            elif FS[i] > 0.7 and FS[i] <= 0.8 and qc1ncs[i] >= 33 and qc1ncs[i] <= 80 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.7 and FS[i] <= 0.8 and qc1ncs[i] > 80 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1690*qc1ncs[i]**-1.46
            elif FS[i] > 0.8 and FS[i] <= 0.9 and qc1ncs[i] >= 33 and qc1ncs[i] <= 60 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.8 and FS[i] <= 0.9 and qc1ncs[i] > 60 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1430*qc1ncs[i]**-1.48
            elif FS[i] > 0.9 and FS[i] <= 1.0 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 64*qc1ncs[i]**-0.93
            elif FS[i] > 1.0 and FS[i] <= 1.1 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 11*qc1ncs[i]**-0.65
            elif FS[i] > 1.1 and FS[i] <= 1.2 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 9.7*qc1ncs[i]**-0.69  
            elif FS[i] > 1.2 and FS[i] <= 1.3 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 7.6*qc1ncs[i]**-0.71
            else:
                epsv[i] = 0.0
                
        sett = [a/100*dz for a in epsv]
        sett2 = np.cumsum(sett)
        sett3 = [np.max(sett2)-i for i in sett2]
        
        # CREATE DATAFRAME ============================================================
        tabel = {"depth_m":z, "elev":elev, "qc_MPa":qc, "fs_MPa": fs, "Rf_%":rf, "gamma_kN/m3":gamma,"sigtot_kPa":sig_tot, 
                 "u_kPa":u0, "sigeff":sig_eff, "Qt":Qt, "Fr":Fr, "Ic":Ic, "delta":delta, "check":check,
                 "rd":rd, "CN":CN, "qc1n":qc1n, "Kc":Kc, "qc1ncs":qc1ncs, "Ksigma":Ksigma, "CRR_m75":CRR_m75, "CRR_m":CRR_m,
                 "CSR_m":CSR_m, "FS":FS, 
                 "F":F, "wz":wz, "F_wz":F_wz, "LPI":INT, "epsv":epsv, "sett_m":sett3}
        
        df = pd.DataFrame(tabel)
        
        # PLOTTING ====================================================================
        ymin = np.min(elev)
        ymax = np.max(elev)
        
        fig, ax = plt.subplots(1, 5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(15,6))
        plt.suptitle('\n'+CPT_label+', LPI = %.2f, Category = %s'%(LPI,category), size=15, x=0.1, y=1.03, horizontalalignment='left')
        ax1 = ax[0]
        ax1.plot(qc,elev,color='black', label="qc")
        ax1.plot([100,100],[0,10],color='green', label="Rf") # this is only dummy
        ax1.plot([0,15],[gwl,gwl],color='blue', linestyle="--", label="gwl", zorder=3)
        ax1.set_ylim(ymin,ymax)
        ax1.set_xlim(0,15.0)
        # ax1.invert_yaxis()
        ax1.minorticks_on()
        ax1.grid(False, which='both')
        ax1.legend(loc=3)
        ax1.set_ylabel("Elev (%s)"%(elev_unit),size=12)
        ax1.set_xlabel("qc (MPa)",size=12)
        ax2 = ax1.twiny()
        ax2.plot(rf,elev,color='green', label="Rf")
        ax2.set_ylim(ymin,ymax)
        ax2.set_xlim(0,30.0)
        ax2.invert_xaxis()
        # ax2.invert_yaxis()
        ax2.set_xlabel("Rf (%)",size=12)
        
        ax3 = ax[1]
        ax3.plot(Ic,elev,color='black',linewidth=1,zorder=3)
        ax3.set_ylim(ymin,ymax)
        ax3.set_xlim(1.0,4.0)
        # ax3.invert_yaxis()
        ax3.add_patch(patches.Rectangle((0,ymin),1.31,ymax-ymin,facecolor='goldenrod'))
        ax3.add_patch(patches.Rectangle((1.31,ymin),0.74,ymax-ymin,facecolor='khaki'))
        ax3.add_patch(patches.Rectangle((2.05,ymin),0.55,ymax-ymin,facecolor='lightsteelblue'))
        ax3.add_patch(patches.Rectangle((2.60,ymin),0.35,ymax-ymin,facecolor='yellowgreen'))
        ax3.add_patch(patches.Rectangle((2.95,ymin),0.65,ymax-ymin,facecolor='olivedrab'))
        ax3.add_patch(patches.Rectangle((3.6,ymin),0.4,ymax-ymin,facecolor='sienna'))
        ax3.yaxis.grid(which="minor")
        ax3.yaxis.grid(which="major")
        ax3.xaxis.grid(which="major")
        ax3.minorticks_on()
        ax3.text(1.05, ymin+1, "gravel - dense sand", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(1.55, ymin+1, "clean sand - silty sand", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(2.22, ymin+1, "silty sand - sandy silt", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(2.65, ymin+1, "clayey silt - silty clay", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(3.2, ymin+1, "silty clay - clay", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(3.65, ymin+1, "peat", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.set_xlabel("Ic",size=12)
        ax3.set_yticks([])
        
        ax4 = ax[2]
        ax4.plot(FS,elev,color="black")
        ax4.plot([1,1],[ymin,ymax],linestyle="--",color="black")
        for i in range(ndata-1):
            if FS[i+1] <= 1.0:
                ax4.fill_betweenx([elev[i],elev[i+1]],[FS[i],FS[i+1]],[1,1],color='salmon')
        ax4.set_ylim(ymin,ymax)
        ax4.set_xlim(0,2)
        # ax4.invert_yaxis()
        ax4.yaxis.grid(which="major")
        ax4.yaxis.grid(which="minor")
        ax4.xaxis.grid(which="major")
        ax4.grid(True)
        # ax4.set_xticks([])
        ax4.set_xlabel("FS",size=12)
        ax4.set_yticks([])
        # ax4.legend(bbox_to_anchor=(1,1), loc="upper left",edgecolor='none')
        
        ax5 = ax[3]
        ax5.plot(INT,elev,color='black')
        ax5.set_ylim(ymin,ymax)
        ax5.set_xlim(0,20.0)
        # ax5.invert_yaxis()
        # ax5.yaxis.grid(which="major")
        ax5.xaxis.grid(which="major")
        ax5.minorticks_on()
        ax5.add_patch(patches.Rectangle((0,ymin),2,(ymax-ymin),facecolor='lightgreen'))
        ax5.add_patch(patches.Rectangle((2,ymin),3,(ymax-ymin),facecolor='yellow'))
        ax5.add_patch(patches.Rectangle((5,ymin),10,(ymax-ymin),facecolor='orange'))
        ax5.add_patch(patches.Rectangle((15,ymin),5,(ymax-ymin),facecolor='red'))
        ax5.text(0.4, ymin+1, "very low", va='bottom', rotation=90, size=13, color="black")
        ax5.text(2.8, ymin+1, "low", va='bottom', rotation=90, size=13, color="black")
        ax5.text(9.4, ymin+1, "high", va='bottom', rotation=90, size=13, color="black")
        ax5.text(17.2, ymin+1, "very high", va='bottom', rotation=90, size=13, color="black")
        ax5.set_xlabel("LPI",size=12)
        
        ax6 = ax[4]
        ax6.plot(sett3,elev,color='black')
        ax6.text(np.max(sett3)-0.02, ymax-(0.02*(ymax-ymin)), "max settlement = %.2f mm"%(np.max(sett3)*1000), va='top', rotation=90, size=10, color="black",zorder=1)
        ax6.set_ylim(ymin,ymax)
        ax6.set_xlabel("induced settlement (m)",size=12)
        ax6.xaxis.grid(which="major")
        ax6.set_yticks([])
        # plt.tight_layout()
        
        return(fig,df,LPI,category)
    
    def LSN(self):
        wb = self.file
        sheet_general = wb['general']
        sheet_cpt = wb['cpt']
        
        CPT_label = sheet_general.cell(2,2).value
        Z = sheet_general.cell(5,2).value
        gwl = Z - self.gwd          
        # self.gwd = Z - gwl
        elev_unit = sheet_general.cell(7,2).value
        
        z, qc, fs, elev = [],[],[],[]
        for i in range(sheet_cpt.max_row - 1):
            z.append(sheet_cpt.cell(i+2,1).value)
            qc.append(np.abs(sheet_cpt.cell(i+2,2).value))
            fs.append(np.abs(sheet_cpt.cell(i+2,3).value))
            elev.append(Z - z[i])                                 
        
        ndata = len(z)
         
        rf = [(i/j*100) for i,j in zip(fs,qc)]
        
        # unit weight (gamma) =========================================================
        gamma = [10*(0.27*np.log10(x)+0.36*np.log10(y/0.1)+1.236) for x,y in zip(rf,qc)]
        
        # vertical stress (sigma) =====================================================
        sig_tot = np.zeros(ndata)
        u0 = np.zeros(ndata)
        dz = z[1]-z[0]
        sig_tot[0] = z[0]*gamma[0]
        for i in range(ndata-1):
            sig_tot[i+1] = (sig_tot[i] + dz*gamma[i+1])
            if z[i+1] > self.gwd:
                u0[i+1] = (u0[i] + dz*10.0)
        sig_eff = sig_tot - u0
        
        # I-SBT (Robertson) ===========================================================
        Qt = [(x*1000-y)/z for x,y,z in zip(qc,sig_tot,sig_eff)]
        Fr = [a*1000/(b*1000-c)*100 for a,b,c in zip(fs,qc,sig_tot)]
        Ic = [np.sqrt((3.47-np.log10(x))**2 + (np.log10(y)+1.22)**2) for x,y in zip(Qt,Fr)]
        
        index = []                  # this is only for plotting purpose
        for i in range(ndata):
            if Ic[i] > 3.6:
                index.append(2)
            elif Ic[i] > 2.95 and Ic[i] <=3.6:
                index.append(3)
            elif Ic[i] > 2.60 and Ic[i] <=2.95:
                index.append(4)
            elif Ic[i] > 2.05 and Ic[i] <=2.60:
                index.append(5)
            elif Ic[i] > 1.31 and Ic[i] <=2.05:
                index.append(6)
            else:
                index.append(7)
        
        # LIQUEFACTION TRIGGERING PROCEDURE ===========================================
        alpha = [1-0.4113*a**0.5+0.04052*a+0.001753*a**1.5 for a in z]
        beta = [1-0.4177*a**0.5+0.05729*a-0.006205*a**1.5+0.00121*a**2 for a in z]
        rd = [a/b for a,b in zip(alpha,beta)]
        
        delta = np.zeros(len(z))
        for i in range(len(z)-1):
            delta[i+1] = np.abs((Ic[i+1]-Ic[i])/(z[i+1]-z[i]))
        
        check = []
        for i in range(len(delta)):
            if Ic[i] > 1.3 and Ic[i] < 3.6 and delta[i] > 2:
                check.append(1)
            else:
                check.append(-1)
        
        CN2 = (100/sig_eff)**0.5
        CN = []
        for j in range(ndata):
            if CN2[j] > 1.7:
                CN.append(1.7)
            else:
                CN.append(CN2[j])
        
        qc1n = [a*b*1000/100 for a,b in zip(CN,qc)]
        
        Kc = []
        for i in range(ndata):
            if Ic[i] <= 1.64:
                Kc.append(1.0)
            else:
                Kc.append(-0.403*Ic[i]**4+5.581*Ic[i]**3-21.63*Ic[i]**2+33.75*Ic[i]-17.88)
        
        qc1ncs = [a*b for a,b in zip(Kc,qc1n)]
        
        # K-SIGMA ---------------------------------------------------------------------
        
        f = 0.7
        Ksigma = [(a/100)**(f-1) for a in sig_eff]
        
        # MAGNITUDE SCALING FACTOR ----------------------------------------------------
        MSF = (self.Mw/7.5)**-2.56
        
        # CYCLIC RESISTANCE RATIO -----------------------------------------------------
        CRR_m75 = []
        for i in range(ndata):
            if qc1ncs[i] < 50:
                CRR_m75.append(0.833*(qc1ncs[i]/1000)+0.05)
            elif qc1ncs[i] >= 50 and qc1ncs[i] < 160:
                CRR_m75.append(93*(qc1ncs[i]/1000)**3+0.08)
            else:
                CRR_m75.append(1.0)
        
        CRR_m = [a*b*MSF for a,b in zip(CRR_m75,Ksigma)]
        
        # CYCLIC STRESS RATIO ---------------------------------------------------------
        CSR_m = [0.65*a/b*self.PGA*c*1.3 for a,b,c in zip(sig_tot,sig_eff,rd)]
        
        # FACTOR OF SAFETY ------------------------------------------------------------
        FS = []
        for i in range(ndata):
            if elev[i] > gwl:
                FS.append(5)
            elif elev[i] < gwl and Ic[i] > 2.6:
                FS.append(5)
            else:
                FS.append(CRR_m[i]/CSR_m[i])
                
        # SETTLEMENT CALCULATION ------------------------------------------------------
        epsv = np.zeros(ndata)
        for i in range(ndata):
            if FS[i] <= 0.5 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.5 and FS[i] <= 0.6 and qc1ncs[i] >= 33 and qc1ncs[i] <= 147 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.5 and FS[i] <= 0.6 and qc1ncs[i] > 147 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 2411*qc1ncs[i]**-1.45
            elif FS[i] > 0.6 and FS[i] <= 0.7 and qc1ncs[i] >= 33 and qc1ncs[i] <= 110 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.6 and FS[i] <= 0.7 and qc1ncs[i] > 110 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1701*qc1ncs[i]**-1.42
            elif FS[i] > 0.7 and FS[i] <= 0.8 and qc1ncs[i] >= 33 and qc1ncs[i] <= 80 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.7 and FS[i] <= 0.8 and qc1ncs[i] > 80 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1690*qc1ncs[i]**-1.46
            elif FS[i] > 0.8 and FS[i] <= 0.9 and qc1ncs[i] >= 33 and qc1ncs[i] <= 60 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 102*qc1ncs[i]**-0.82
            elif FS[i] > 0.8 and FS[i] <= 0.9 and qc1ncs[i] > 60 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 1430*qc1ncs[i]**-1.48
            elif FS[i] > 0.9 and FS[i] <= 1.0 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 64*qc1ncs[i]**-0.93
            elif FS[i] > 1.0 and FS[i] <= 1.1 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 11*qc1ncs[i]**-0.65
            elif FS[i] > 1.1 and FS[i] <= 1.2 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 9.7*qc1ncs[i]**-0.69  
            elif FS[i] > 1.2 and FS[i] <= 1.3 and qc1ncs[i] >= 33 and qc1ncs[i] <= 200 and z[i] > self.gwd and check[i]<0.0:
                epsv[i] = 7.6*qc1ncs[i]**-0.71
            else:
                epsv[i] = 0.0
                
        sett = [a/100*dz for a in epsv]
        sett2 = np.cumsum(sett)
        sett3 = [np.max(sett2)-i for i in sett2]
        
        # LSN CALCULATION -------------------------------------------------------------
        
        fungsi_LSN = [1000*dz*a/(b*100) for a,b in zip(epsv,z)]
        fungsi_LSN2 = np.cumsum(fungsi_LSN)
        LSN = np.max(fungsi_LSN2)
        # print(LSN)
        
        if LSN < 10:
            category = "None"
        elif LSN >= 10 and LSN < 20:
            category = "Minor"
        elif LSN >= 20 and LSN < 30:
            category = "Moderate"  
        elif LSN >= 30 and LSN < 40:
            category = "Severe"  
        elif LSN >= 40:
            category = "Extreme" 
        
        # CREATE DATAFRAME ============================================================
        tabel = {"depth_m":z, "elev":elev, "qc_MPa":qc, "fs_MPa": fs, "Rf_%":rf, "gamma_kN/m3":gamma,"sigtot_kPa":sig_tot, 
                 "u_kPa":u0, "sigeff":sig_eff, "Qt":Qt, "Fr":Fr, "Ic":Ic, "delta":delta, "check":check,
                 "rd":rd, "CN":CN, "qc1n":qc1n, "Kc":Kc, "qc1ncs":qc1ncs, "Ksigma":Ksigma, "CRR_m75":CRR_m75, "CRR_m":CRR_m,
                 "CSR_m":CSR_m, "FS":FS, 
                 "LSN":fungsi_LSN, "epsv":epsv, "sett_m":sett3}
        
        df = pd.DataFrame(tabel)
        
        # PLOTTING ====================================================================
        ymin = np.min(elev)
        ymax = np.max(elev)
        
        fig, ax = plt.subplots(1, 5, gridspec_kw={'width_ratios': [1, 1, 1, 1, 1]}, figsize=(15,6))
        plt.suptitle('\n'+CPT_label+', LSN = %.2f, Category = %s'%(LSN,category), size=15, x=0.1, y=1.03, horizontalalignment='left')
        ax1 = ax[0]
        ax1.plot(qc,elev,color='black', label="qc")
        ax1.plot([100,100],[0,10],color='green', label="Rf") # this is only dummy
        ax1.plot([0,15],[gwl,gwl],color='blue', linestyle="--", label="gwl", zorder=3)
        ax1.set_ylim(ymin,ymax)
        ax1.set_xlim(0,15.0)
        # ax1.invert_yaxis()
        ax1.minorticks_on()
        ax1.grid(False, which='both')
        ax1.legend(loc=3)
        ax1.set_ylabel("Elev (%s)"%(elev_unit),size=12)
        ax1.set_xlabel("qc (MPa)",size=12)
        ax2 = ax1.twiny()
        ax2.plot(rf,elev,color='green', label="Rf")
        ax2.set_ylim(ymin,ymax)
        ax2.set_xlim(0,30.0)
        ax2.invert_xaxis()
        # ax2.invert_yaxis()
        ax2.set_xlabel("Rf (%)",size=12)
        
        ax3 = ax[1]
        ax3.plot(Ic,elev,color='black',linewidth=1,zorder=3)
        ax3.set_ylim(ymin,ymax)
        ax3.set_xlim(1.0,4.0)
        # ax3.invert_yaxis()
        ax3.add_patch(patches.Rectangle((0,ymin),1.31,ymax-ymin,facecolor='goldenrod'))
        ax3.add_patch(patches.Rectangle((1.31,ymin),0.74,ymax-ymin,facecolor='khaki'))
        ax3.add_patch(patches.Rectangle((2.05,ymin),0.55,ymax-ymin,facecolor='lightsteelblue'))
        ax3.add_patch(patches.Rectangle((2.60,ymin),0.35,ymax-ymin,facecolor='yellowgreen'))
        ax3.add_patch(patches.Rectangle((2.95,ymin),0.65,ymax-ymin,facecolor='olivedrab'))
        ax3.add_patch(patches.Rectangle((3.6,ymin),0.4,ymax-ymin,facecolor='sienna'))
        ax3.yaxis.grid(which="minor")
        ax3.yaxis.grid(which="major")
        ax3.xaxis.grid(which="major")
        ax3.minorticks_on()
        ax3.text(1.05, ymin+1, "gravel - dense sand", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(1.55, ymin+1, "clean sand - silty sand", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(2.22, ymin+1, "silty sand - sandy silt", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(2.65, ymin+1, "clayey silt - silty clay", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(3.2, ymin+1, "silty clay - clay", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.text(3.65, ymin+1, "peat", va='bottom', rotation=90, size=10, color="white",zorder=1)
        ax3.set_xlabel("Ic",size=12)
        ax3.set_yticks([])
        
        ax4 = ax[2]
        ax4.plot(FS,elev,color="black")
        ax4.plot([1,1],[ymin,ymax],linestyle="--",color="black")
        for i in range(ndata-1):
            if FS[i+1] <= 1.0:
                ax4.fill_betweenx([elev[i],elev[i+1]],[FS[i],FS[i+1]],[1,1],color='salmon')
        ax4.set_ylim(ymin,ymax)
        ax4.set_xlim(0,2)
        # ax4.invert_yaxis()
        ax4.yaxis.grid(which="major")
        ax4.yaxis.grid(which="minor")
        ax4.xaxis.grid(which="major")
        ax4.grid(True)
        # ax4.set_xticks([])
        ax4.set_xlabel("FS",size=12)
        ax4.set_yticks([])
        # ax4.legend(bbox_to_anchor=(1,1), loc="upper left",edgecolor='none')
        
        ax5 = ax[3]
        ax5.plot(fungsi_LSN2,elev,color='black',label='LSN')
        ax5.plot([10,10],[ymin,ymax],color='silver',linestyle='--')
        ax5.plot([20,20],[ymin,ymax],color='silver',linestyle='--')
        ax5.plot([30,30],[ymin,ymax],color='silver',linestyle='--')
        ax5.plot([40,40],[ymin,ymax],color='silver',linestyle='--')
        ax5.text(LSN-0.1*LSN, ymin+1, "category = %s"%(category), va='bottom', rotation=90, size=10, color="black",zorder=1)
        ax5.set_ylim(ymin,ymax)
        ax5.set_xlabel("LSN (kN/m)",size=12)
        ax5.set_yticks([])
        ax5.set_xlim(0,LSN*1.1)
        
        ax6 = ax[4]
        ax6.plot(sett3,elev,color='black')
        ax6.text(np.max(sett3)-0.02, ymax-(0.02*(ymax-ymin)), "max settlement = %.2f mm"%(np.max(sett3)*1000), va='top', rotation=90, size=10, color="black",zorder=1)
        ax6.set_ylim(ymin,ymax)
        ax6.set_xlabel("induced settlement (m)",size=12)
        ax6.xaxis.grid(which="major")
        ax6.set_yticks([])
        # plt.tight_layout()
        
        return(fig,df,LSN,category)