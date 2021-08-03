## load packages---------------------------------------------------------------------------
library(DESeq2)  
library(reshape2)
library(dplyr)
library(ggplot2)

## Read data-------------------------------------------------------------------------------
info.first <- read.csv("background_first.csv", header=TRUE, row.names=1)
data.first <- read.csv("readcount_first.csv", header=TRUE, row.names=1)
info.second <- read.csv("background_second.csv", header=TRUE, row.names=1)
data.second <- read.csv("readcount_second.csv", header=TRUE, row.names=1)

info <- data.frame (rbind(info.first, info.second))
data <- data.frame (cbind(data.first, data.second))

all(rownames(info) == colnames(data)) #TRUE
sum <- data.frame(row.names(info), info$Targets.Detected)
colnames(sum) <- c("Sample", "Detected")

## Density plot (without cleaning)---------------------------------------------------------                                                    
pseudo <- log2(data + 1)
df <- melt(pseudo, variable.name = "Samples")
df <- data.frame(df)
ggplot(df, aes(x = value, colour = Samples)) + 
  ylim(c(0, 0.5)) +
  geom_line(stat="density") +
  ylab("Density") +
  xlab(expression(paste(log[2],"(normalized count + 1)"))) +
  theme_bw() +
  theme(legend.position = "none", 
        axis.title= element_text(size=20),
        axis.text = element_text(size=15)) 

## Data cleaning---------------------------------------------------------------------------
lowTargets <- sum[sum$Detected<0.2, 1] #Targets detected 20% threshold
data.filtered <- data[, !(colnames(data) %in% lowTargets)]
filtered_samples <- length(data[1,]) - length(lowTargets)

index <- as.matrix(info)
sample_info.filtered <- as.data.frame(
  index[!(rownames(info) %in% lowTargets),])
colnames(sample_info.filtered) <- colnames(info)
all(rownames(sample_info.filtered) == colnames(data.filtered)) #TRUE

## Library size normalization--------------------------------------------------------------
d2 <- DESeqDataSetFromMatrix(countData=data.filtered, 
                             colData=sample_info.filtered,
                             design=~Condition)
colData(d2)$condition <- relevel(colData(d2)$Condition, ref="Ctrl") #base for DE comparison
d2 <- estimateSizeFactors(d2)

## Remove genes with no count in **% samples-----------------------------------------------
accept2 <- ceiling(filtered_samples*0.9) #detection rate threshold
idx2 <- rowSums(counts(d2) > 0) >= accept2 #read count threshold
d2 <- d2[idx2,]
d.norm2 <- counts(d2, normalized=TRUE)

## Density plot (post-cleaning)------------------------------------------------------------
log.norm2 <- as.data.frame(log2(d.norm2 + 1))
df.norm2 <- melt(log.norm2, variable.name = "Samples")
df.norm2 <- data.frame(df.norm2)

ggplot(df.norm2, aes(x = value, colour = Samples)) + 
  ylim(c(0, 0.5)) +
  xlim(c(0, 15)) + 
  geom_line(stat="density") + 
  ylab("Density") +
  xlab(expression(paste(log[2],"(normalized count + 1)"))) +
  theme_bw() +
  theme(legend.position = "none", 
        axis.title= element_text(size=20),
        axis.text = element_text(size=15))

      
# END -----------------------------------------------------------------------------------

