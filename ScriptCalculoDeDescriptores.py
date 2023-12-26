# -*- coding: utf-8 -*-
"""Step-02-Generacion-descriptores-DatasetPeptidos.ipynb

INPUT: Lista de secuencias, Lista de descriptores 
OUTPUT: DataFrame con secuencias y descriptores correspondientes

"""

import pandas as pd
import numpy as np
from modlamp_modified.descriptors import PeptideDescriptor,GlobalDescriptor #modlamp
import time

#input_seq = df_novel_pep.novel_sequences[0:1000].tolist()
def get_all_descriptors():
    return tuple( ['Y', 'V', 'W', 'T', 'R', 'S', 'P', 'Q', 'N', 'L', 'M', 'K', 'H', 'I', 'F', 'G', 'D', 'E', 'C', 'A', 'Length', 'MW', 'Charge', 'ChargeDensity', 'pI', 'InstabilityInd', 'Aromaticity', 'AliphaticInd', 'BomanInd', 'HydrophRatio', 'AASI_1_mean', 'ABHPRK_1_mean', 'argos_1_mean', 'bulkiness_1_mean', 'charge_phys_1_mean', 'charge_acid_1_mean', 'cougar_1_mean', 'eisenberg_1_mean', 'Ez_1_mean', 'flexibility_1_mean', 'grantham_1_mean', 'gravy_1_mean', 'hopp-woods_1_mean', 'ISAECI_1_mean', 'janin_1_mean', 'levitt_alpha_1_mean', 'MSS_1_mean', 'PPCALI_1_mean', 'seeva_v2_1_mean', 'seeva_v3_1_mean', 'seeva_v4_1_mean', 'seeva_v5_1_mean', 'seeva_v6_1_mean', 'seeva_v7_1_mean', 'seeva_v8_1_mean', 'seeva_v9_1_mean', 'seeva_v10_1_mean', 'seeva_v11_1_mean', 'seeva_v12_1_mean', 'vsw_v1_1_mean', 'vsw_v2_1_mean', 'vsw_v3_1_mean', 'vsw_v4_1_mean', 'vsw_v5_1_mean', 'vsw_v6_1_mean', 'vsw_v7_1_mean', 'vsw_v8_1_mean', 'vsw_v9_1_mean', 'svger_v2_1_mean', 'svger_v3_1_mean', 'svger_v4_1_mean', 'svger_v5_1_mean', 'svger_v6_1_mean', 'svger_v8_1_mean', 'svger_v9_1_mean', 'svger_v11_1_mean', 'svrg_v2_1_mean', 'svrg_v3_1_mean', 'svrg_v5_1_mean', 'svrg_v6_1_mean', 'svrg_v7_1_mean', 'svrg_v8_1_mean', 'svrg_v9_1_mean', 'svrg_v10_1_mean', 'svrg_v12_1_mean', 'svrg_v16_1_mean', 'hesh_v4_1_mean', 'hesh_v9_1_mean', 'ANDN920101_1_mean', 'BEGF750102_1_mean', 'BROC820101_1_mean', 'BUNA790103_1_mean', 'BURA740102_1_mean', 'CHAM820102_1_mean', 'CHAM830102_1_mean', 'CHAM830103_1_mean', 'CHAM830104_1_mean', 'CHAM830107_1_mean', 'CHOP780204_1_mean', 'CHOP780205_1_mean', 'CHOP780206_1_mean', 'CHOP780207_1_mean', 'CHOP780212_1_mean', 'CHOP780214_1_mean', 'CHOP780215_1_mean', 'DAYM780201_1_mean', 'FASG760102_1_mean', 'FASG760104_1_mean', 'FAUJ880105_1_mean', 'FAUJ880107_1_mean', 'FAUJ880108_1_mean', 'FAUJ880112_1_mean', 'FINA910101_1_mean', 'GEIM800102_1_mean', 'GEIM800103_1_mean', 'GEIM800106_1_mean', 'HOPA770101_1_mean', 'ISOY800102_1_mean', 'ISOY800106_1_mean', 'KARP850103_1_mean', 'LEVM760103_1_mean', 'MAXF760103_1_mean', 'NAKH900102_1_mean', 'NAKH900104_1_mean', 'OOBM850103_1_mean', 'OOBM850105_1_mean', 'PALJ810111_1_mean', 'PALJ810113_1_mean', 'PALJ810115_1_mean', 'PONP800105_1_mean', 'PRAM820101_1_mean', 'PRAM820103_1_mean', 'QIAN880103_1_mean', 'QIAN880104_1_mean', 'QIAN880114_1_mean', 'QIAN880116_1_mean', 'QIAN880117_1_mean', 'QIAN880123_1_mean', 'QIAN880124_1_mean', 'QIAN880128_1_mean', 'QIAN880129_1_mean', 'QIAN880139_1_mean', 'RACS820101_1_mean', 'RACS820103_1_mean', 'RACS820107_1_mean', 'RACS820111_1_mean', 'RACS820112_1_mean', 'RACS820113_1_mean', 'RICJ880101_1_mean', 'RICJ880103_1_mean', 'RICJ880104_1_mean', 'RICJ880105_1_mean', 'RICJ880106_1_mean', 'RICJ880107_1_mean', 'RICJ880110_1_mean', 'RICJ880113_1_mean', 'RICJ880114_1_mean', 'RICJ880116_1_mean', 'RICJ880117_1_mean', 'ROSM880103_1_mean', 'SUEM840102_1_mean', 'TANS770102_1_mean', 'TANS770108_1_mean', 'VASM830101_1_mean', 'VASM830103_1_mean', 'WERD780102_1_mean', 'WERD780103_1_mean', 'WERD780104_1_mean', 'WOLS870103_1_mean', 'ZIMJ680101_1_mean', 'AURR980101_1_mean', 'AURR980102_1_mean', 'AURR980103_1_mean', 'AURR980104_1_mean', 'AURR980105_1_mean', 'AURR980106_1_mean', 'AURR980107_1_mean', 'AURR980118_1_mean', 'AURR980120_1_mean', 'PARS000102_1_mean', 'NADH010107_1_mean', 'KOEP990102_1_mean', 'FUKS010101_1_mean', 'AVBF000108_1_mean', 'AVBF000109_1_mean', 'WILM950101_1_mean', 'WILM950104_1_mean', 'GEOR030106_1_mean', 'AGK', 'CKI', 'RR', 'YGGG', 'LSGL', 'RG', 'YGGY', 'PRP', 'LGGG', 'GV', 'GT', 'GS', 'GR', 'IAG', 'GG', 'GF', 'GC', 'GGYG', 'GA', 'GL', 'GK', 'GI', 'IPC', 'KAA', 'LAK', 'GLGG', 'GGLG', 'CKIT', 'GAGK', 'LLSG', 'LKK', 'FLP', 'LSG', 'SCK', 'LLS', 'GETC', 'VLG', 'GKLL', 'LLG', 'KCKI', 'VGK', 'CSC', 'TKKC', 'GCS', 'GKA', 'IGK', 'GESC', 'KVCY', 'KKL', 'KKI', 'KKC', 'LGGL', 'GLL', 'CGE', 'GGYC', 'GLLS', 'GLF', 'AKK', 'GKAA', 'ESCV', 'GLP', 'CGES', 'PCGE', 'FL', 'CGET', 'GLW', 'KGAA', 'KAAL', 'GGY', 'GGG', 'IKG', 'LKG', 'GGL', 'CK', 'GTC', 'CG', 'SKKC', 'CS', 'CR', 'KC', 'AGKA', 'KA', 'KG', 'LKCK', 'SCKL', 'KK', 'KI', 'KN', 'KL', 'SK', 'KV', 'SL', 'SC', 'SG', 'AAA', 'VAK', 'AAL', 'AAK', 'GGGG', 'KNVA', 'GGGL', 'GYG', 'LG', 'LA', 'LL', 'LK', 'LS', 'LP', 'GCSC', 'TC', 'GAA', 'AA', 'VA', 'VC', 'AG', 'VG', 'AI', 'AK', 'VL', 'AL', 'TPGC', 'IK', 'IA', 'IG', 'YGG', 'LGK', 'CSCK', 'GYGG', 'LGG', 'KGA'])

def DescriptoresGlobalesCalculo(input_seq,tupla_deseados): 
    #DESCRIPTORES GLOBALES DEL PAQUETE MODLAMP
    
    #Nombre de los descriptores 
    global_descriptor_names=['Length','MW','Charge','ChargeDensity','pI','InstabilityInd','Aromaticity',
    'AliphaticInd','BomanInd','HydrophRatio']
    
    #Calculo de descriptores
    print("Calculando descripptores globales de los peptidos ...")
    GlobalD = GlobalDescriptor(input_seq)     
    GlobalD.calculate_all(ph=7.4, amide=False)  
    D=GlobalD.descriptor                         # Array con descriptores 
    return pd.DataFrame(GlobalD.descriptor, columns= global_descriptor_names)



#DESCRIPTORES PEPTIDICOS DEL PAQUETE MODLAMP

def DescriptoresPeptidicosCalculo(input_seq,tupla_deseados,ventana=[1], modalidades=['mean']): 
    print("Calculando descripptores peptidicos ...")
    #Nombre de los descriptores 
    name_descriptors=('AASI','ABHPRK','argos','bulkiness','charge_phys','charge_acid',
    'cougar','eisenberg', 'Ez', 'flexibility', 'grantham','gravy','hopp-woods',
    'ISAECI','janin','kytedoolittle','levitt_alpha','MSS','MSW','pepArc',
    'pepcats','polarity','PPCALI','refractivity','t_scale','TM_tend','z3','z5',
    'seeva_v1','seeva_v2','seeva_v3','seeva_v4','seeva_v5','seeva_v6','seeva_v7',
    'seeva_v8','seeva_v9','seeva_v10','seeva_v11','seeva_v12','vsw_v1','vsw_v2',
    'vsw_v3','vsw_v4','vsw_v5','vsw_v6','vsw_v7','vsw_v8','vsw_v9', 'svger_v1',
    'svger_v2','svger_v3','svger_v4','svger_v5','svger_v6','svger_v7','svger_v8','svger_v9','svger_v10',
    'svger_v11','svrg_v1','svrg_v2','svrg_v3','svrg_v4','svrg_v5','svrg_v6','svrg_v7',
    'svrg_v8','svrg_v9','svrg_v10','svrg_v11','svrg_v12','svrg_v13','svrg_v14','svrg_v15','svrg_v16',
    'hesh_v1','hesh_v2','hesh_v3','hesh_v4','hesh_v5','hesh_v6','hesh_v7','hesh_v8','hesh_v9',
    'hesh_v10','hesh_v11','hesh_v12', 'ANDN920101', 'ARGP820101','ARGP820102','ARGP820103','BEGF750101',
    'BEGF750102','BEGF750103','BHAR880101','BIGC670101','BIOV880101','BIOV880102','BROC820101','BROC820102',
    'BULH740101','BULH740102','BUNA790101','BUNA790102','BUNA790103','BURA740101','BURA740102','CHAM810101',
    'CHAM820101','CHAM820102','CHAM830101','CHAM830102','CHAM830103','CHAM830104','CHAM830105','CHAM830106',
    'CHAM830107','CHAM830108','CHOC750101','CHOC760101','CHOC760102','CHOC760103','CHOC760104','CHOP780101',
    'CHOP780201','CHOP780202','CHOP780203','CHOP780204','CHOP780205','CHOP780206','CHOP780207','CHOP780208',
    'CHOP780209','CHOP780210','CHOP780211','CHOP780212','CHOP780213','CHOP780214','CHOP780215','CHOP780216',
    'CIDH920101','CIDH920102','CIDH920103','CIDH920104','CIDH920105','COHE430101','CRAJ730101','CRAJ730102',
    'CRAJ730103','DAWD720101','DAYM780101','DAYM780201','DESM900101','DESM900102','EISD840101','EISD860101',
    'EISD860102','EISD860103','FASG760101','FASG760102','FASG760103','FASG760104','FASG760105','FAUJ830101',
    'FAUJ880101','FAUJ880102','FAUJ880103','FAUJ880104','FAUJ880105','FAUJ880106','FAUJ880107','FAUJ880108',
    'FAUJ880109','FAUJ880110','FAUJ880111','FAUJ880112','FAUJ880113','FINA770101','FINA910101','FINA910102',
    'FINA910103','FINA910104','GARJ730101','GEIM800101','GEIM800102','GEIM800103','GEIM800104','GEIM800105',
    'GEIM800106','GEIM800107','GEIM800108','GEIM800109','GEIM800110','GEIM800111','GOLD730101','GOLD730102',
    'GRAR740101','GRAR740102','GRAR740103','GUYH850101','HOPA770101','HOPT810101','HUTJ700101','HUTJ700102',
    'HUTJ700103','ISOY800101','ISOY800102','ISOY800103','ISOY800104','ISOY800105','ISOY800106','ISOY800107',
    'ISOY800108','JANJ780101','JANJ780102','JANJ780103','JANJ790101','JANJ790102','JOND750101','JOND750102',
    'JOND920101','JOND920102','JUKT750101','JUNJ780101','KANM800101','KANM800102','KANM800103','KANM800104',
    'KARP850101','KARP850102','KARP850103','KHAG800101','KLEP840101','KRIW710101','KRIW790101','KRIW790102',
    'KRIW790103','KYTJ820101','LAWE840101','LEVM760101','LEVM760102','LEVM760103','LEVM760104','LEVM760105',
    'LEVM760106','LEVM760107','LEVM780101','LEVM780102','LEVM780103','LEVM780104','LEVM780105','LEVM780106',
    'LEWP710101','LIFS790101','LIFS790102','LIFS790103','MANP780101','MAXF760101','MAXF760102','MAXF760103',
    'MAXF760104','MAXF760105','MAXF760106','MCMT640101','MEEJ800101','MEEJ800102','MEEJ810101','MEEJ810102',
    'MEIH800101','MEIH800102','MEIH800103','MIYS850101','NAGK730101','NAGK730102','NAGK730103','NAKH900101',
    'NAKH900102','NAKH900103','NAKH900104','NAKH900105','NAKH900106','NAKH900107','NAKH900108','NAKH900109',
    'NAKH900110','NAKH900111','NAKH900112','NAKH900113','NAKH920101','NAKH920102','NAKH920103','NAKH920104',
    'NAKH920105','NAKH920106','NAKH920107','NAKH920108','NISK800101','NISK860101','NOZY710101','OOBM770101',
    'OOBM770102','OOBM770103','OOBM770104','OOBM770105','OOBM850101','OOBM850102','OOBM850103','OOBM850104',
    'OOBM850105','PALJ810101','PALJ810102','PALJ810103','PALJ810104','PALJ810105','PALJ810106','PALJ810107',
    'PALJ810108','PALJ810109','PALJ810110','PALJ810111','PALJ810112','PALJ810113','PALJ810114','PALJ810115',
    'PALJ810116','PARJ860101','PLIV810101','PONP800101','PONP800102','PONP800103','PONP800104','PONP800105',
    'PONP800106','PONP800107','PONP800108','PRAM820101','PRAM820102','PRAM820103','PRAM900101','PRAM900102',
    'PRAM900103','PRAM900104','PTIO830101','PTIO830102','QIAN880101','QIAN880102','QIAN880103','QIAN880104',
    'QIAN880105','QIAN880106','QIAN880107','QIAN880108','QIAN880109','QIAN880110','QIAN880111','QIAN880112',
    'QIAN880113','QIAN880114','QIAN880115','QIAN880116','QIAN880117','QIAN880118','QIAN880119','QIAN880120',
    'QIAN880121','QIAN880122','QIAN880123','QIAN880124','QIAN880125','QIAN880126','QIAN880127','QIAN880128',
    'QIAN880129','QIAN880130','QIAN880131','QIAN880132','QIAN880133','QIAN880134','QIAN880135','QIAN880136',
    'QIAN880137','QIAN880138','QIAN880139','RACS770101','RACS770102','RACS770103','RACS820101','RACS820102',
    'RACS820103','RACS820104','RACS820105','RACS820106','RACS820107','RACS820108','RACS820109','RACS820110',
    'RACS820111','RACS820112','RACS820113','RACS820114','RADA880101','RADA880102','RADA880103','RADA880104',
    'RADA880105','RADA880106','RADA880107','RADA880108','RICJ880101','RICJ880102','RICJ880103','RICJ880104',
    'RICJ880105','RICJ880106','RICJ880107','RICJ880108','RICJ880109','RICJ880110','RICJ880111','RICJ880112',
    'RICJ880113','RICJ880114','RICJ880115','RICJ880116','RICJ880117','ROBB760101','ROBB760102','ROBB760103',
    'ROBB760104','ROBB760105','ROBB760106','ROBB760107','ROBB760108','ROBB760109','ROBB760110','ROBB760111',
    'ROBB760112','ROBB760113','ROBB790101','ROSG850101','ROSG850102','ROSM880101','ROSM880102','ROSM880103',
    'SIMZ760101','SNEP660101','SNEP660102','SNEP660103','SNEP660104','SUEM840101','SUEM840102','SWER830101',
    'TANS770101','TANS770102','TANS770103','TANS770104','TANS770105','TANS770106','TANS770107','TANS770108',
    'TANS770109','TANS770110','VASM830101','VASM830102','VASM830103','VELV850101','VENT840101','VHEG790101',
    'WARP780101','WEBA780101','WERD780101','WERD780102','WERD780103','WERD780104','WOEC730101','WOLR810101',
    'WOLS870101','WOLS870102','WOLS870103','YUTK870101','YUTK870102','YUTK870103','YUTK870104','ZASB820101',
    'ZIMJ680101','ZIMJ680102','ZIMJ680103','ZIMJ680104','ZIMJ680105','AURR980101','AURR980102','AURR980103',
    'AURR980104','AURR980105','AURR980106','AURR980107','AURR980108','AURR980109','AURR980110','AURR980111',
    'AURR980112','AURR980113','AURR980114','AURR980115','AURR980116','AURR980117','AURR980118','AURR980119',
    'AURR980120','ONEK900101','ONEK900102','VINM940101','VINM940102','VINM940103','VINM940104','MUNV940101',
    'MUNV940102','MUNV940103','MUNV940104','MUNV940105','WIMW960101','KIMC930101','MONM990101','BLAM930101',
    'PARS000101','PARS000102','KUMS000101','KUMS000102','KUMS000103','KUMS000104','TAKK010101','FODM020101',
    'NADH010101','NADH010102','NADH010103','NADH010104','NADH010105','NADH010106','NADH010107','MONM990201',
    'KOEP990101','KOEP990102','CEDJ970101','CEDJ970102','CEDJ970103','CEDJ970104','CEDJ970105','FUKS010101',
    'FUKS010102','FUKS010103','FUKS010104','FUKS010105','FUKS010106','FUKS010107','FUKS010108','FUKS010109',
    'FUKS010110','FUKS010111','FUKS010112','AVBF000101','AVBF000102','AVBF000103','AVBF000104','AVBF000105',
    'AVBF000106','AVBF000107','AVBF000108','AVBF000109','YANJ020101','MITS020101','TSAJ990101','TSAJ990102',
    'COSI940101','PONP930101','WILM950101','WILM950102','WILM950103','WILM950104','KUHL950101','GUOD860101',
    'JURD980101','BASU050101','BASU050102','BASU050103','SUYM030101','PUNT030101','PUNT030102','GEOR030101',
    'GEOR030102','GEOR030103','GEOR030104','GEOR030105','GEOR030106','GEOR030107','GEOR030108','GEOR030109',
    'ZHOH040101','ZHOH040102','ZHOH040103','BAEK050101','HARY940101','PONJ960101','DIGM050101','WOLR790101',
    'OLSK800101','KIDA850101','GUYH850102','GUYH850103','GUYH850104','GUYH850105','ROSM880104','ROSM880105',
    'JACR890101','COWR900101','BLAS910101','CASG920101','CORJ870101','CORJ870102','CORJ870103','CORJ870104')
    
    tupla_deseados_transf = tuple([s[:-7] for s in tupla_deseados])

    # Performing intersection operation 
    descriptores_comunes = tuple(set(name_descriptors) & set(tupla_deseados_transf))
    
    
    #Parametros para calcular los descriptores moleculares 
    #ventana=[1]  #Numero de aminoacidos por ventana, Ex: [1,3,4,7,11,18,29,47,76,123]
    #modalidades=["mean"] #Forma promediar todas las ventanas calulcadas, Ex: ["mean","max"]
    
    
    print('Numero de calculos : ',len(descriptores_comunes)*len(ventana)*len(modalidades)*len(input_seq))
    
    
    #Bucle for: Calcula los descriptores con cada ventana y cada modalidad y lo anexa al dataset
    
    from alive_progress import alive_bar
    
    #Bucle for: Calcula los descriptores con cada ventana y cada modalidad y lo anexa al dataset
    t1=time.time()
    
    df_secuencias = pd.DataFrame({'Secuencias': input_seq})
    
    for k in ventana:
      for l in modalidades:
          with alive_bar(len(descriptores_comunes),theme="classic") as bar:
            for i in descriptores_comunes:
                desc = PeptideDescriptor(list(input_seq),i)
                desc.calculate_global(window=k, modality=l)
                name_variable=i+"_"+str(k)+"_"+l
                df_secuencias=pd.concat([df_secuencias, pd.DataFrame(desc.descriptor,columns=[name_variable])], axis=1)
                
                bar.title(f"Ventana:{k}, Modalidad: {l}")
                bar()
              
    t2=time.time()
    print(t2-t1)
    
    df_secuencias = df_secuencias.drop('Secuencias', axis=1)
    return df_secuencias


def DescriptoresAACalculo(input_seq): 
    print("Calculando descripptores de frecuencia de aminoacidos ...")
    #Descriptores de frecuencia absoluta de aminoacidos DEL PAQUETE MODLAMP
    #['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    freq_names=['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y']
    ModFreq=PeptideDescriptor(list(input_seq))
    ModFreq.count_aa(scale='absolute')
    freq_aa = ModFreq.descriptor
    
    df_freq_aa = pd.DataFrame(freq_aa, columns=freq_names)
    return df_freq_aa



#Descriptores de frecuencia de n-gramas de aminoacidos DEL PAQUETE MODLAMP
from modlamp_modified.core import ngrams_apd


def DescriptoresKmerCalculo(input_seq): 
    print("Calculando descripptores de Kmers de aa de los peptidos ...")
    apd_n_grams=ngrams_apd()
    apd_n_grams = pd.Series(apd_n_grams)
    n_grams=[]
    
    for i in apd_n_grams:
        n_g_sec=[]
        for k in input_seq:
            n=k.count(i)
            n_g_sec.append(n)
        n_grams.append(n_g_sec)

    
    a=pd.DataFrame(n_grams, index=apd_n_grams).transpose()
    return a

def CalcularDescriptores(input_seq,descriptores_deseados=get_all_descriptors(),ventana=[1], modalidades=['mean']):
    t1=time.time()

    Descriptor_AA = DescriptoresAACalculo(input_seq)
    Descriptor_Glob = DescriptoresGlobalesCalculo(input_seq,descriptores_deseados)
    Descriptor_Pep = DescriptoresPeptidicosCalculo(input_seq,descriptores_deseados,ventana=[1], modalidades=['mean'])
    Descriptor_Kmer = DescriptoresKmerCalculo(input_seq)
    
    df_descriptores = pd.concat([Descriptor_AA, Descriptor_Glob, Descriptor_Pep ,Descriptor_Kmer ],axis=1)
    print('Tiempo de calculo: ',time.time()-t1)
    df_descriptores = df_descriptores.loc[:, ~df_descriptores.columns.duplicated()]
    
    df_descriptores = df_descriptores [list(descriptores_deseados)]
    df_descriptores.index = input_seq
    
    return df_descriptores


#aa = CalcularDescriptores(input_seq,descriptores_deseados,ventana=[1], modalidades=['mean'])















#################################################################################
#                       CORREGIR DESCRIPTORES DE AMINOACIDOS
#                                        CTMNR
#################################################################################