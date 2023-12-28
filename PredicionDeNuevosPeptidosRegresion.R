library(reticulate)
os <- import("os")
os$listdir(".")

# Instala y carga el paquete Biostrings
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("Biostrings")


# Cargar librerias
packages_load <- c("tidyverse",  "readxl", "recipes", "rstatix",
                   "caret", "caretEnsemble", "doParallel", "gbm","readr","Biostrings" )

for (i in packages_load) {
  library(i, character.only  = TRUE)
}


# Lee el archivo FASTA
fasta_file <- "ejemplo.fa"
sequences <- readDNAStringSet(fasta_file)

# Convierte a dataframe
peptides <- data.frame(names = names(sequences), novel_sequences = as.character(sequences), stringsAsFactors = FALSE)

# Muestra el dataframe
print(peptides)


CalculoDescriptoresPy <- function(input) { #Input = dataframe 
  DS <- import("ScriptCalculoDeDescriptores")
  data = DS$CalcularDescriptores(input$novel_sequences)
}

peptide_descriptors <- CalculoDescriptoresPy(peptides) 
peptide_descriptors$hoopwoods_1_mean <- peptide_descriptors$'hopp-woods_1_mean'


# Lista de carpetas donde se encuentran los archivos .rda (MODELOS Y PREPROCESSING)
carpetas <- list.dirs( "./Models/AMP_Regresion", recursive = FALSE)


# Inicializa una lista para almacenar las predicciones
predicciones_lista <- list()


for (carpeta in carpetas) {
  # Carga el archivo .rda desde la carpeta
  load(paste0(carpeta, "/models_logMIC.rda"))
  load(paste0(carpeta, "/FinalPreprocessing_algoritm.rda"))
  
  data <- bake(rec_train2, new_data = peptide_descriptors)
  
  # Accede al modelo 'rf' en el archivo .rda
  modelo_rf <- models_logMIC$rf
  
  # Genera predicciones con el modelo 'rf'
  print(paste("Predicciones: ",carpeta))
  predicciones <- predict(modelo_rf, data)  # Reemplaza 'nuevos_datos' con tus datos de entrada
  
  # Almacena las predicciones en la lista
  predicciones_lista[[carpeta]] <- predicciones
}

predicciones <- as.data.frame(predicciones_lista) %>%
  mutate(Sequences = peptides$names )


