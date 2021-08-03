## load packages---------------------------------------------------------------------------
library(DESeq2)  
library(reshape2)
library(dplyr)
library(ggplot2)
library(gplots)
library(ggbeeswarm)

## Read data-------------------------------------------------------------------------------
info <- read.csv("background_first.csv", header=TRUE, row.names=1)
data <- read.csv("readcount_first.csv", header=TRUE, row.names=1)
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

## Differential expression analysis--------------------------------------------------------
d2 <- estimateDispersions(d2)
d2 <- nbinomLRT(d2, full = ~Condition, reduced = ~1)
tmp <- results(d2)
p.value <- tmp$pvalue
q.value <- tmp$padj

# DEG labeling and export csv--------------------------------------------------------------
res <- cbind(rownames(d2), d.norm2, as.data.frame(tmp)[,1:4], p.value, q.value)
res <- res[order(res$q.value),]
results = as.data.frame(mutate(as.data.frame(res),
                               DE_test=ifelse((res$log2FoldChange)>log2(1) & # FC threshold
                                                res$q.value<0.25, #FDR threshold
                                              "DEG_up",
                                              ifelse(-(res$log2FoldChange)>log2(1) & # FC threshold
                                                       res$q.value<0.25, #FDR threshold
                                                     "DEG_down", "non-DEG"))),
                        row.names(res))
write.csv(results, "DEanalysis_result_first.csv", row.names=F)
       
# volcano plot----------------------------------------------------------------------------
ggplot(results, aes(log2FoldChange, -log10(q.value)))+
  geom_point(aes(col=DE_test), alpha=0.5, size=2)+
  scale_color_manual(values=c("blue", "red", "gray70"))+
  scale_x_continuous(breaks=seq(-6, 6, by=2), limits=c(-6, 6)) +
  xlab(expression(Log[2](FoldChange))) +
  ylab(expression(-Log[10](FDR))) +
  theme_bw() +
  theme(legend.position = "none")

# PCA-------------------------------------------------------------------------------------
vst <- varianceStabilizingTransformation(d2, blind=TRUE)
vst.mat <- assay(vst)
vst.mat[vst.mat==0] <- NA
vst.mat <- scale(vst.mat, center=TRUE, scale=TRUE)
Pvars <- rowVars(vst.mat)
select <- order(Pvars, decreasing=TRUE)[seq_len(min(1000, length(Pvars)))] #top1000 variables
pt3 <- min(200/filtered_samples, 4)
pca <- prcomp(t(vst.mat[select, ]), scale=F)
summary(pca)$importance

# PCA 2D-plot-----------------------------------------------------------------------------
pca.df <- data.frame(pca$x[,1:3])
pca.df$label = colData(vst)$Condition
colnames(pca.df) <- c("v1","v2", "v3", "Label")
ggplot(data=pca.df, aes(x=v1, y=v2, color=Label)) +
  geom_point(size = 3) +
  theme_bw() +
  theme (axis.title = element_text(face='plain'),
         legend.position = "top") + 
  xlab("PC1 (12.2%)") + ylab("PC2 (10.8%)") +
  guides(color=guide_legend(title=NULL))
ggplot(data=pca.df, aes(x=v1, y=v3, color=Label)) +
  geom_point(size = 3) +
  theme_bw() +
  theme (axis.title = element_text(face='plain'),
         legend.position = "top") + 
  xlab("PC1 (12.2%)") + ylab("PC3 (8.42%)") +
  guides(color=guide_legend(title=NULL))
ggplot(data=pca.df, aes(x=v2, y=v3, color=Label)) +
  geom_point(size = 3) +
  theme_bw() +
  theme (axis.title = element_text(face='plain'),
         legend.position = "top") + 
  xlab("PC2 (10.8%)") + ylab("PC3 (8.42%)") +
  guides(color=guide_legend(title=NULL))

# Heatmap of OXPHOS genes----------------------------------------------------------------
OXPHOSgene <- c("ATP5B", "ATP5E", "ATP5O", "COX4I1", "COX8A",
                "NDUFA4L2", "NDUFB11", "NDUFB2", "NDUFB8", "NDUFS5",
                "UBE2L3", "UQCR11", "UQCRB", "UQCRC1", "UQCRFS1", "UQCRH")
d.norm.OX <- d.norm2[OXPHOSgene,]
d.norm.OX <- d.norm.OX[,order(colnames(d.norm.OX), decreasing = FALSE)] # sort samples
d.norm.OX <- data.frame(d.norm.OX[,8:20], d.norm.OX[,1:7])# sort by group (Ctrl and PD)
z.score <- t(scale(t(log2(d.norm.OX+1))))
z.score[z.score>4] <- 4
z.score[z.score<(-4)] <- (-4)
z.score2 <- melt(z.score)
colnames(z.score2) <- c("Gene","Sample","z_score")
ggplot(z.score2, aes(x = Sample, y = Gene, fill = z_score)) +
  geom_tile(color = "white", size = 0.1) +
  scale_fill_gradientn("z-score", colours = bluered(20),limits=c(-4,4), na.value="gray70") +
  theme_classic() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5),
        axis.title = element_blank(),
        legend.position = "none")

# Barplot of shared genes----------------------------------------------------------------
Sharegene <- c("ANXA1", "AQP3", "ATP6V0C", "BHLHE40", "CCL3", 
               "CCNI", "CXCR4", "EGR2", "EMP1", "GABARAPL1", 
               "KRT16", "POLR2L", "RHOA", "RNASEK", "SERINC1", "SERPINB4", "SNORA24")
condition <- rep(c("PD","Ctrl"), c(7,13))
d.norm.Sh <- data.frame(condition, t(log2(d.norm2[Sharegene,]+1)))
d.norm.Shm <- melt(d.norm.Sh)
ggplot(d.norm.Shm, aes(x=variable, y=value, fill=condition)) +
  geom_bar(position = 'dodge', stat = 'summary') +
  geom_quasirandom(dodge.width = 0.9, cex = 0.5, alpha=0.5, shape = 1) +
  stat_summary(fun.min = function(x)mean(x) - sd(x), fun.max = function(x)mean(x) + sd(x),
               geom = "errorbar", color = "black",
               position = position_dodge(width = 0.9), width = 0.2, show.legend = FALSE) +
  xlab("") +
  ylab(expression(paste(log[2],"(normalized count + 1)"))) +
  ylim(0,20) +
  theme_bw() +
  theme(axis.text.x= element_text(angle=80, hjust=1),
        legend.title=element_blank())

# Barplot of known PD genes--------------------------------------------------------------
PDgene <- c("CHCHD2", "FBXO7", "GAK", "GBA", 
            "GCH1", "LRRK2", "PARK7", "PINK1", "VPS35")
condition <- rep(c("PD","Ctrl"), c(7,13))
d.norm.PD <- data.frame(condition, t(log2(d.norm2[PDgene,]+1)))
d.norm.PDm <- melt(d.norm.PD)
ggplot(d.norm.PDm, aes(x=variable, y=value, fill=condition)) +
  geom_bar(position = 'dodge', stat = 'summary') +
  geom_quasirandom(dodge.width = 0.9, cex = 0.5, alpha=0.5, shape = 1) +
  stat_summary(fun.min = function(x)mean(x) - sd(x), fun.max = function(x)mean(x) + sd(x),
               geom = "errorbar", color = "black",
               position = position_dodge(width = 0.9), width = 0.2, show.legend = FALSE) +
  xlab("") +
  ylab(expression(paste(log[2],"(normalized count + 1)"))) +
  ylim(0,20) +
  theme_bw() +
  theme(axis.text.x= element_text(angle=80, hjust=1),
        legend.title=element_blank())

# END -----------------------------------------------------------------------------------