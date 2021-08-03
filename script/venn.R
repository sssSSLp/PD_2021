# read csv
n15w <- read.csv("n15_wald.csv", header=TRUE)
n50w <- read.csv("n50_wald.csv", header=TRUE)
n15l <- read.csv("n15_lrt.csv", header=TRUE)
n50l <- read.csv("n50_lrt.csv", header=TRUE)
n15r <- read.csv("n15_rpm.csv", header=TRUE)
n50r <- read.csv("n50_rpm.csv", header=TRUE)
# as.character
n15w <- as.character(n15w[,1])
n50w <- as.character(n50w[,1])
n15l <- as.character(n15l[,1])
n50l <- as.character(n50l[,1])
n15r <- as.character(n15r[,1])
n50r <- as.character(n50r[,1])



library(VennDiagram)
list <- list(n15=n15r, n50=n50r)
venn.diagram(list, filename="rpm.png", fill=c(2,3), alpha=0.4, lty=1)
intersect(n15r,n50r)
marge <- c(intersect(n15r,n50r))
write.csv(marge, "intersect_rpm.csv")

list2 <- list(n15_LRT=n15l, n15_Log2RPM=n15r, n50_LRT=n50l, n50_Log2RPM=n50r)
venn.diagram(list2, filename="all.png", fill=c(2,3,4,5), alpha=0.4, lty=1)
intersect(intersect(intersect(n15l,n15r), n50l), n50r) # pickup shared genes
