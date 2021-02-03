from NuPyCEE import omega as o
from matplotlib import pyplot as plt 
import numpy as np

o1=o.omega(galaxy='milky_way',mgal=1.5e12,imf_type='kroupa')
o2=o.omega(galaxy='milky_way',mgal=1.5e12,imf_type='salpeter')

iso_ratio=['C-12/C-13','O-16/O-18','O-16/O-17']

for i in range(0,len(iso_ratio)):
    o1.plot_iso_ratio(yaxis=iso_ratio[i],color='r',label='Kroupa IMF')
    o2.plot_iso_ratio(yaxis=iso_ratio[i],color='b',label='Salpeter IMF')
    ii=str(i)
    plt.savefig('/home/ziyi/PostGraduate/GCE/N3/1_salpeter_and_kroupa_mw_sfh/'+ii+'.pdf',bbox_inches='tight')
    plt.title(iso_ratio[i])
    plt.close()

elements=['O','Mg','Ca','Ti','Si','C','N']

for i in range(0,len(elements)):
    o1.plot_mass(specie=elements[i],label='Kroupa IMF',color='r')
    o2.plot_mass(specie=elements[i],label='Salpeter IMF',color='b')
    plt.savefig('/home/ziyi/PostGraduate/GCE/N3/1_salpeter_and_kroupa_mw_sfh/'+elements[i]+'.pdf',bbox_inches='tight')
    plt.title(elements[i])
    plt.close()

ele_ratio=['[O/Fe]','[Mg/Fe]','[Ca/Fe]','[Ti/Fe]','[Si/Fe]']

for i in range(0,len(ele_ratio)):
    o1.plot_spectro(yaxis=ele_ratio[i],label='Kroupa IMF',color='r')
    o2.plot_spectro(yaxis=ele_ratio[i],label='Salpeter IMF',color='b')
    ii=str(i)
    plt.ylim(-0.5,0.2)
    plt.savefig('/home/ziyi/PostGraduate/GCE/N3/1_salpeter_and_kroupa_mw_sfh/ele_ratio'+ii+'.pdf',bbox_inches='tight')
    plt.title(ele_ratio[i])
    plt.close()

for i in range(0,len(ele_ratio)):
    o1.plot_spectro(yaxis=ele_ratio[i],xaxis='[Fe/H]',label='Kroupa IMF',color='r')
    o2.plot_spectro(yaxis=ele_ratio[i],xaxis='[Fe/H]',label='Salpeter IMF',color='b')
    ii=str(i)
    plt.savefig('/home/ziyi/PostGraduate/GCE/N3/1_salpeter_and_kroupa_mw_sfh/ele_ratio_to_Fe'+ii+'.pdf',bbox_inches='tight')
    plt.title(ele_ratio[i])
    plt.close()



