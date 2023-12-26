# -*- coding: utf-8 -*-
"""
PREDICCION DE PEPTIDOS 

INPUT: DATAFRAME DE PEPTIDOS CON SUS NOMBRES PROVISIONALES y SUS SECUENCIAS

OUTPUT: DATAFRAME PEPTIDSO: NOMBRE, SEQ, CLASIFIFACION DE AMP o NO AMP y DOMINIO DE APLICABILIDAD

@author: Eliezer Bonifacio
"""

import pandas as pd
from joblib import load
from ScriptCalculoDeDescriptores import CalcularDescriptores


def GenerateDescriptors(data_input):
    """
    Parameters
    ----------
    data_input : pd.DataFrame
        Column 1("names"): Names of sequences
        Column 2("novel_sequences"): sequences, only natural aa

    Returns
    -------
    NovelPeptidesDescript : pd.DataFrame 
        Descriptors of data_input (3.5 sec for 100 aa)
    """

    secuencias = data_input.novel_sequences.tolist()
    NovelPeptidesDescript = CalcularDescriptores(secuencias,ventana=[1], modalidades=['mean'])
    return NovelPeptidesDescript


prefijos = ["VsAll/PeptidesVsAll_",
            "VsGramN/PeptidesVsGramN_",
            "VsGramP/PeptidesVsGramP_",
            "VsEscherichia/PeptidesVsEscherichia_",
            "VsPseudomonas/PeptidesVsPseudomonas_",
            "VsStaphylococcus/PeptidesVsStaphylococcus_",
            "VsBacillus/PeptidesVsBacillus_"
    ]





def GenerateClasificationPrediction(data_input, descriptores=None):
    """
    Generates clasification predicctions of peptides in data_input

    Parameters
    ----------
    data_input : pd.DataFrame
        Column 1("names"): Names of sequences
        Column 2("novel_sequences"): sequences, only natural aa
        
    descriptores : pd.DataFrame, optional
        Descriptors generated, the function can generate it automaticaly

    Returns
    -------
    pd.DataFrame
        DataFrame with clasification prediccions and determination of they aplicability domian  
    """
    Out1 = data_input.copy()
    if descriptores == None:
        Descriptores = GenerateDescriptors(data_input)
        print("Descriptores calculados \n")
        
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬") 
    print("Realizando predicciones ...") 
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬") 
    for i in prefijos:
        print(i)
        Preprocessor = load("Models/AMP_Clasificacion/"+ i+"FinalPreprocessing.joblib") #Rute to final preprocessor
        
        x_var = pd.DataFrame(Preprocessor.transform(Descriptores),
                             columns= Preprocessor.get_feature_names_out())
        
        
        #Cargamos modelos de clasififacion de AMP: 
        
        ruta_modelo = 'Models/AMP_Clasificacion/'+ i+'ModelRF.joblib' #Rute to model "RF"
        Modelo = load(ruta_modelo)
        
        AMP_proba =  Modelo.predict_proba(x_var)[:,1]
        
        
        #Determinamos el dominio de aplicabilidad
        
        Processor_AD = load("Models/AMP_Clasificacion/"+ i+"pipeApDomain.joblib")
        
        AMP_AD = Processor_AD.decision_function(x_var)
    
        PredName = "Prediccion_"+i[:i.find("/")] 
        ADName = "AD_"+i[:i.find("/")] 
        #Name +Secuencia + AMP Proba + AD 
        
        Out1[PredName] = AMP_proba
        Out1[ADName] = AMP_AD
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬") 
    print("Predicciones realizadas satisfactoriamente ...") 
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬") 
    
    return pd.DataFrame(Out1)


