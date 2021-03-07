from NuPyCEE import omega as o
from NuPyCEE import sygma as s
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import math
from sympy import *
import time

start = time.time()

# Set the parameters
# a1 = -0.3
# a2 = -1.3
# a3 = -0.9
imf_type = 'kroupa'
burst_time = '1e9'
bt=1e9
file_name = '3_outflow/burst_time_'+burst_time+'/'

# Set the SFH for the model
os.remove('NuPyCEE/stellab_data/burst_time_'+burst_time+'.txt')
n = Symbol('n')
t = Symbol('t')
f = (n+1.8)*bt*0.5 + 1.8*(13e9-bt)-5e10
n = solve(f,n)[0]
sfh = np.zeros((3,2),dtype=np.float)
sfh[0][1] = n
sfh[1][0] = bt
sfh[1][1] = 1.8
sfh[2][0] = 13e9
sfh[2][1] = 1.8
np.savetxt('NuPyCEE/stellab_data/burst_time_'+burst_time+'.txt',sfh,fmt='%f',delimiter=' ')

# The elements and isotopes 
# ratios=['C-12/C-13','O-16/O-18','O-18/O-17']
elements=['H','O','Mg','Ca','Ti','Si','C','N','Fe','S']
all_iso=['H-1', 'H-2', 'He-3', 'He-4','Li-7', 'B-11','C-12','C-13','N-14','N-15','O-16','O-17','O-18','F-19','Ne-20','Ne-21','Ne-22','Na-23','Mg-24','Mg-25','Mg-26','Al-27','Si-28','Si-29','Si-30','P-31','Pb-206','Pb-207','S-32','S-33','S-34','S-36','Cl-35','Cl-37','Ar-36','Ar-38','Ar-40','K-39','K-40','K-41','Ca-40','Ca-42','Ca-43','Ca-44','Ca-46','Ca-48','Sc-45','Ti-46','Ti-47','Ti-48','Ti-49','Ti-50','V-50','V-51','Cr-50','Cr-52','Cr-53','Cr-54','Mn-55','Fe-54','Fe-56','Fe-57','Fe-58','Co-59','Ni-58','Ni-60','Ni-61','Ni-62','Ni-64','Cu-63','Cu-65','Zn-64','Zn-66','Zn-67','Zn-68','Zn-70','Ga-69','Ga-71','Ge-70','Ge-72','Ge-73','Ge-74','Ge-76','As-75','Se-74','Se-76','Se-77','Se-78','Se-80','Se-82','Br-79','Br-81','Kr-78','Kr-80','Kr-82','Kr-83','Kr-84','Kr-86','Rb-85','Rb-87','Sr-84','Sr-86','Sr-87','Sr-88','Y-89','Zr-90','Zr-91','Zr-92','Zr-94','Zr-96','Nb-93','Mo-92','Mo-94','Mo-95','Mo-96','Mo-97','Mo-98','Mo-100','Ru-96','Ru-98','Ru-99','Ru-100','Ru-101','Ru-102','Ru-104','Rh-103','Pd-102','Pd-104','Pd-105','Pd-106','Pd-108','Pd-110','Ag-107','Ag-109','Cd-106','Cd-108','Cd-110','Cd-111','Cd-112','Cd-113','Cd-114','Cd-116','In-113','In-115','Sn-112','Sn-114','Sn-115','Sn-116','Sn-117','Sn-118','Sn-119','Sn-120','Sn-122','Sn-124','Sb-121','Sb-123','Te-120','Te-122','Te-123','Te-124','Te-125','Te-126','Te-128','Te-130','I-127','Xe-124','Xe-126','Xe-128','Xe-129','Xe-130','Xe-131','Xe-132','Xe-134','Xe-136','Cs-133','Ba-130','Ba-132','Ba-134','Ba-135','Ba-136','Ba-137','Ba-138','La-138','La-139','Ce-136','Ce-138','Ce-140','Ce-142','Pr-141','Nd-142','Nd-143','Nd-144','Nd-145','Nd-146','Nd-148','Nd-150','Sm-144','Sm-147','Sm-148','Sm-149','Sm-150','Sm-152','Sm-154','Eu-151','Eu-153','Gd-152','Gd-154','Gd-155','Gd-156','Gd-157','Gd-158','Gd-160','Tb-159','Dy-156','Dy-158','Dy-160','Dy-161','Dy-162','Dy-163','Dy-164','Ho-165','Er-162','Er-164','Er-166','Er-167','Er-168','Er-170','Tm-169','Yb-168','Yb-170','Yb-171','Yb-172','Yb-173','Yb-174','Yb-176','Lu-175','Lu-176','Hf-174','Hf-176','Hf-177','Hf-178','Hf-179','Hf-180','Ta-180','Ta-181','W-180','W-182','W-183','W-184','W-186','Re-185','Re-187','Os-184','Os-186','Os-187','Os-188','Os-189','Os-190','Os-192','Ir-191','Ir-193','Pt-190','Pt-192','Pt-194','Pt-195','Pt-196','Pt-198','Au-197','Hg-196','Hg-198','Hg-199','Hg-200','Hg-201','Hg-202','Hg-204','Tl-203','Tl-205','Pb-204','Pb-208','Bi-209'] 

# isotopes=['O-16','O-17','O-18','C-12','C-13','S-32','S-33','S-34','S-36','Si-28','Si-29','Si-30','Ti-46','Ti-47','Ti-48','Ti-49','Ti-50','Ti-44']


# Run the OMEGA
o1=o.omega(imf_type=imf_type,DM_evolution=True,m_DM_0=1e12,stellar_mass_0=5e10,sfh_file='stellab_data/milky_way_data/sfh_mw_cmr01.txt')
o2=o.omega(imf_type=imf_type,DM_evolution=True,sfh_file='stellab_data/burst_time_'+burst_time+'.txt',m_DM_0=1e12,stellar_mass_0=5e10)

# Plot the outflow rate
o1.plot_outflow_rate(label = imf_type+', with normal star formation history',color='r')
o2.plot_outflow_rate(label = imf_type+', with star burst time = '+ burst_time,color='b')
plt.savefig(file_name+"outflow.pdf",bbox_inches='tight')
plt.close()

# Plot the stellar metallicity distribution function
# o1.plot_mdf(label = imf_type+', with normal star formation history',color='r')
# o2.plot_mdf(label = imf_type+', with star burst time = '+ burst_time,color='b')
# plt.savefig(file_name+"mdf.pdf",bbox_inches='tight')
# plt.close()

# Get the mass of the outflow
o1_time,o1_outflow=o1.plot_outflow_rate(return_x_y=True)
o2_time,o2_outflow=o2.plot_outflow_rate(return_x_y=True)
plt.close()
o1_time=np.delete(o1_time,0)
o1_outflow=np.delete(o1_outflow,0)
o2_time=np.delete(o2_time,0)
o2_outflow=np.delete(o2_outflow,0)
# print(o1_time)
# print(o1_outflow)
# plt.plot(o1_time,o1_outflow,color='r')
# plt.savefig(file_name+'outflow_p.pdf')
# plt.close()

# Get the mass of the total mass of the gas
o1_x2,o1_t=o1.plot_totmasses(mass='gas',return_x_y=True)
o2_x2,o2_t=o2.plot_totmasses(mass='gas',return_x_y=True)
plt.close()
o1_x2=np.delete(o1_x2,-1)
o1_t=np.delete(o1_t,-1)
o2_x2=np.delete(o2_x2,-1)
o2_t=np.delete(o2_t,-1)
# print(o1_x2)
 
# Get the mass of elements in the ISM
for e in elements:
    o1_x,o1_m=o1.plot_mass(specie=e,return_x_y=True)
    o2_x,o2_m=o2.plot_mass(specie=e,return_x_y=True)
    plt.close()
    o1_x=np.delete(o1_x,-1)
    o1_m=np.delete(o1_m,-1)
    o2_x=np.delete(o2_x,-1)
    o2_m=np.delete(o2_m,-1)
    # print(o1_x)
    # plt.plot(o1_x,o1_r,color='r')
    # plt.savefig(file_name+'00'+e+'.pdf')
    # plt.close()
    
    # Make sure the time is the same
    if len(o1_x2)!=len(o1_time):
        print('The length of time is different!')
    else:
        if not (o1_x2==o1_time).all():
            print('The timestep of two function is different!')
            print(o1_x)
            print(o1_time)
        else:
            # Calculate the mass of the elements in the outflow
            o1_n=np.zeros(len(o1_m))
            o2_n=np.zeros(len(o2_m))
            for i in range(len(o1_m)):
                o1_n[i]=o1_m[i]/o1_t[i]*o1_outflow[i]
                o2_n[i]=o2_m[i]/o2_t[i]*o2_outflow[i]
            # print(o1_n)
            plt.plot(o1_x,o1_n,label = imf_type+', with normal star formation history',color='r')
            plt.plot(o2_x,o2_n,label = imf_type+', with normal star formation history',color='b')
            # plt.yscale('log')
            plt.xlabel('Age[yr]')
            plt.ylabel(r'the mass of '+e+' in ourflow[$M_{\odot}$]')
            plt.legend()
            plt.title(e)
            plt.savefig(file_name+e+'_outflow.pdf',bbox_inches='tight')
            plt.close()

# Test if the total mass is the same as the mass of the outflow
tot1=np.zeros((len(o1_x2),len(all_iso)))
tot2=np.zeros((len(o2_x2),len(all_iso)))
a=0
for iso in all_iso:
    o1_a,o1_b=o1.plot_mass(specie=iso,return_x_y=True)
    o2_a,o2_b=o2.plot_mass(specie=iso,return_x_y=True)
    plt.close()
    #print(o1_a)
    o1_a=np.delete(o1_a,-1)
    o1_b=np.delete(o1_b,-1)
    o2_a=np.delete(o2_a,-1)
    o2_b=np.delete(o2_b,-1)
    # print(o1_a)
    # plt.plot(o1_1,o1_r,color='r')
    # plt.savefig(file_name+'00'+e+'.pdf')
    # plt.close()
    if len(o1_x2)!=len(o1_time):
        print('The length of time is different!')
    else:
        if not (o1_x2==o1_time).all():
            print('The timestep of two function is different!')
            print(o1_1)
            print(o1_time)
        else:
            # o1_n=np.zeros(len(o1_2))
            # o2_n=np.zeros(len(o2_2))
            for i in range(len(o1_a)):
                tot1[i][a]=o1_b[i]/o1_t[i]*o1_outflow[i]
                tot2[i][a]=o2_b[i]/o2_t[i]*o2_outflow[i]
            a+=1
            print(a)
            # print(o1_n)
            # plt.plot(o1_1,o1_n,label = imf_type+', with normal star formation history',color='r')
            # plt.plot(o2_1,o2_n,label = imf_type+', with normal star formation history',color='b')
            # # plt.yscale('log')
            # plt.xlabel('Age[yr]')
            # plt.ylabel(r'the mass of '+iso+' in ourflow[$M_{\odot}$]')
            # plt.legend()
            # plt.title(e)
            # plt.savefig(file_name+e+'_outflow.pdf',bbox_inches='tight')
            # plt.close()

outflow1=0
outflow2=0
for j in range(a):
    outflow1+=tot1[-1][j]
    outflow2+=tot2[-1][j]

if outflow1==o1_outflow[-1]:
    print('o1 is perfect!')
else:
    print(outflow1-o1_outflow[-1])

if outflow2==o2_outflow[-1]:
    print('o2 is perfect!')
else:
    print(outflow2-o2_outflow[-1])


end = time.time()
print('Complete! Running time is '+str(end-start)+'s!')
