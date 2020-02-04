getwd()
setwd("//Users//nikhilviswanath//Documents//Rdata")
file.exists("ect_data.txt")

reduction_data=read.table("ect_data.txt",header = T,sep="\t")

### creating training and validation data###

sample_size <- floor(0.5*nrow(reduction_data))
set.seed(123)
train_ind <- sample(seq_len(nrow(reduction_data)),size = sample_size)
train_data <- reduction_data[train_ind,]
testing_data <- reduction_data[-train_ind,]

####Performing PCA and FA ######
colnames(reduction_data)

reduction_data.pca=train_data[c("attitude1_01","attitude1_02","attitude1_03","attitude1_04","intent1_01","intent1_02","intent1_03","intent1_04","peruse01","peruse02","peruse03","peruse04","satis01","satis02","satis03","satis04")]

pcamodel_reduc = princomp(reduction_data.pca,cor=FALSE)
pcamodel_reduc$sdev^2

##### PCA shows that 4 distinct components exsists based on Eigenvalues greater than 1####

###Creating a scree plot to verify the results#####

pcamodel_reduc = princomp(reduction_data.pca, cor=FALSE)
plot(pcamodel_reduc, main="Screeplot of attitude, intent, peruse and satisfaction")

####scree plot also indcating 4 components###

###performing a factor analysis####

redcution_data.fa=factanal(~attitude1_01+attitude1_02+attitude1_03+attitude1_04+intent1_01+intent1_02+intent1_03+intent1_04+peruse01+peruse02+peruse03+peruse04+satis01+satis02+satis03+satis04,
                           factors=4,
                           rotation="varimax",
                           scores = "none",
                           data=reduction_data)

redcution_data.fa
####################################
### Performing Cluster analysis ####
#####################################

file.exists("car.test.frame.txt")

car_data=read.table("car.test.frame.txt", header=T,sep="\t")
summary(car_data)

##looking at plots to understand the clusters###

par(mfrow=c(1,3))

car_data[,6:8] <- scale(car_data[,6:8])
car_data[,1] <- scale(car_data[,1])
car_data[,3:4] <- scale(car_data[,3:4])

car_data <-na.omit(car_data)

##K means using 4 clusters####
km_4=kmeans(data.frame(car_data$Reliability,car_data$Mileage,car_data$Weight,car_data$Disp.,car_data$HP),4)
km_4
km_6=kmeans(data.frame(car_data$Reliability,car_data$Mileage,car_data$Weight,car_data$Disp.,car_data$HP),6)
km_6

#####Agglomerative analysis####

km_dist = dist(car_data[,c("Price","Reliability","Mileage","Weight","Disp.","HP")])
km_clust = hclust(km_dist)
plot(km_clust, main="Agglomerative Clustering")


