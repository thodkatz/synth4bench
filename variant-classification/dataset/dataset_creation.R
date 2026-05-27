# required libraries
library(data.table)
library(ggplot2)
library(patchwork)
library(scales)
library(ggtext)
library(ggridges)
library(ggdist)
library(colorspace)
library(shadowtext)
library(stringr)

# data <- fread("C:/Users/sfragkouli/Desktop/files_prin/dataset.tsv")
# varscan = data[which(Caller=="VarScan")]
# summary(varscan$AF)

indels <- fread("C:/Users/sfragkouli/Desktop/files_prin_new/all_indels.tsv")
indels$Coverage <- indels$Dataset |> basename() |> str_split_i("_", 1)
indels$Read_length <- indels$Dataset |> basename() |> str_split_i("_", 2)

colnames(indels) = c("Dataset", "POS", "REF", "ALT", "DP", "AD",         
                     "AF", "mut", "Class", "Indel category", "Caller",
                     "Coverage", "Read_length")

indels$`AF Deviation` = NA
indels$`Variant type` = "Indel"

indels_new = indels[, c("Dataset", "Coverage", "Read_length",     
                        "POS", "REF", "ALT", "DP", "AD", "AF",  "AF Deviation",        
                        "mut", "Caller",  "Indel category", "Variant type", "Class"      
                        )]

rm(indels)

noise <- fread("C:/Users/sfragkouli/Desktop/files_prin_new/all_Noise.tsv")
noise$Coverage <- noise$Dataset |> basename() |> str_split_i("_", 1)
noise$Read_length <- noise$Dataset |> basename() |> str_split_i("_", 2)
colnames(noise) = c("Dataset", "POS", "REF", "ALT", "DP", 
                    "AD", "AF", "mut", "Class", "Caller", 
                    "Coverage", "Read_length")

noise$`Indel category` = NA
noise$`AF Deviation` = NA
noise$`Variant type` = "Noise"

noise_new = noise[, c("Dataset", "Coverage", "Read_length",     
                        "POS", "REF", "ALT", "DP", "AD", "AF",  "AF Deviation",        
                        "mut", "Caller",  "Indel category", "Variant type", "Class"      
)]


rm(noise)

TVs <- fread("C:/Users/sfragkouli/Desktop/files_prin_new/all_TVs.tsv")
TVs$Coverage <- TVs$Dataset |> basename() |> str_split_i("_", 1)
TVs$Read_length <- TVs$Dataset |> basename() |> str_split_i("_", 2)

TVs$`Caller REF`= NULL
TVs$`Caller ALT` = NULL
TVs$`Caller DP` = NULL
TVs$`Caller AF` = NULL



colnames(TVs) = c("Dataset", "POS", "REF", "ALT", "DP", 
  "AF",  "Class", "AF Deviation", "Caller", "Coverage", "Read_length" )


TVs$`Indel category` = NA
TVs$AD = NA
TVs$mut = paste(TVs$POS, 
                TVs$REF, 
                TVs$ALT, sep = ":")

TVs$`Variant type` = "TV"

TVs_new = TVs[, c("Dataset", "Coverage", "Read_length",     
                      "POS", "REF", "ALT", "DP", "AD", "AF", "AF Deviation",       
                      "mut", "Caller",  "Indel category", "Variant type", "Class"       
)]

rm(TVs)


dataset = rbind(TVs_new, noise_new, indels_new)


fwrite(
  dataset, "C:/Users/sfragkouli/Desktop/files_prin_new/dataset.tsv",
  
  row.names = FALSE, quote = FALSE, sep = "\t"
)



