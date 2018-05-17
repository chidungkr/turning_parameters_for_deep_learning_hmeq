

# Load dữ liệu: 

rm(list = ls())
library(tidyverse)
library(magrittr)

hmeq <- read.csv("D:/Teaching/data_science_banking/hmeq/hmeq.csv")

# Viết một số hàm xử lí số liệu thiếu và dán lại nhãn: 
thay_na_mean <- function(x) {
  tb <- mean(x, na.rm = TRUE)
  x[is.na(x)] <- tb
  return(x)
}


name_job <- function(x) {
  x %<>% as.character()
  ELSE <- TRUE
  quan_tam <- c("Mgr", "Office", "Other", "ProfExe", "Sales", "Self")
  case_when(!x %in% quan_tam ~ "Other", 
            ELSE ~ x)
}


name_reason <- function(x) {
  ELSE <- TRUE
  x %<>% as.character()
  case_when(!x %in% c("DebtCon", "HomeImp") ~ "Unknown", 
            ELSE ~ x)
}

label_rename <- function(x) {
  case_when(x == 1 ~ "BAD", 
            x == 0 ~ "GOOD")
}


my_scale01 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}


# Xử lí số liệu thiếu và dán nhãn lại: 
df <- hmeq %>% 
  mutate_if(is.numeric, thay_na_mean) %>% 
  mutate_at("REASON", name_reason) %>% 
  mutate_at("JOB", name_job) %>% 
  mutate(BAD = label_rename(BAD)) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.numeric, my_scale01)

# Chuẩn bị dữ liệu: 
y <- "BAD"
x <- setdiff(names(df), y)

# Load gói h2o cho Deep Learning: 
library(h2o)
h2o.init(nthreads = 6, max_mem_size = "12G")

# Chuyển hóa dữ liệu về h2o Frame: 
hmeq_hf <- df %>% as.h2o()

# Chuẩn bị train và test data: 

# Phân chia dữ liệu: 
splits <- h2o.splitFrame(hmeq_hf, c(0.5), seed = 1234) 
train <- splits[[1]]
test <- splits[[2]]

#-----------------------------------------------------
#   Model 1: Dùng 2 layers, mỗi layer 20 neutrons
#-----------------------------------------------------

dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            hidden = c(20, 20),
                            stopping_metric = "misclassification", 
                            seed = 1)


#-----------------------------------------------------
# Model 2: Dùng 2 layers, mỗi layer 20 neutrons
# nhưng epochs = 50  (mặc định là 10). Nếu tăng
# option này thì có thể dẫn đến overfit. Để hạn 
# chế overfit có thể đặt stopping_rounds = 5. 
# Nếu không đặt stopping_rounds = 0. 
#-----------------------------------------------------


dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            epochs = 50,
                            hidden = c(20, 20),
                            stopping_rounds = 0,
                            stopping_metric = "misclassification", 
                            seed = 1)



#--------------------------------
#   Model 3: Phức tạp hơn chút 
#--------------------------------


dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            epochs = 50,
                            hidden = c(200, 200),
                            nfolds = 5,                            
                            score_interval = 1,                    
                            stopping_rounds = 5,                   
                            stopping_metric = "misclassification", 
                            stopping_tolerance = 0.001,             
                            seed = 1)

# So sánh ba mô hình và nhận thấy ngay  mô hình thứ
# 3 phân loại rất tót nhóm hồ sơ xấu (BAD): 

lapply(list(dl_fit1, dl_fit2, dl_fit3), 
       function(x) {h2o.performance(x, newdata = test)})

# Soi nhanh mô hình thứ 3: 
plot(dl_fit3, 
     timestep = "epochs", 
     metric = "classification_error")


# Điều này  gợi ý rằng nên tăng epochs lên: 
dl_fit4 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            epochs = 1000,
                            hidden = c(200, 200, 200),
                            nfolds = 5,                            
                            score_interval = 1,                    
                            stopping_rounds = 5,                   
                            stopping_metric = "misclassification", 
                            stopping_tolerance = 0.001,             
                            seed = 1)


# Xem qua kết quả: 
h2o.performance(dl_fit4, test)

# Hoặc sử dụng hình ảnh xem quá trình huấn luyện: 

plot(dl_fit4, 
     timestep = "epochs", 
     metric = "classification_error")


cv_models <- sapply(dl_fit4@model$cross_validation_models, 
                    function(i) h2o.getModel(i$name))

plot(cv_models[[1]], 
     timestep = "epochs", 
     metric = "classification_error")

#----------------------------
#    Turning Papameters
#----------------------------

activation_opt <- c("Rectifier", "Maxout", "Tanh")
l1_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)
l2_opt <- c(0, 0.00001, 0.0001, 0.001, 0.01)

hyper_params <- list(activation = activation_opt, l1 = l1_opt, l2 = l2_opt)
search_criteria <- list(strategy = "RandomDiscrete", max_runtime_secs = 600)

splits <- h2o.splitFrame(train, ratios = 0.5, seed = 1)

dl_grid <- h2o.grid("deeplearning", 
                    x = x, 
                    y = y,
                    grid_id = "dl_grid",
                    training_frame = splits[[1]],
                    validation_frame = splits[[2]],
                    epochs = 500, 
                    seed = 1,
                    hidden = c(200, 200, 200),
                    hyper_params = hyper_params,
                    search_criteria = search_criteria)

dl_gridperf <- h2o.getGrid(grid_id = "dl_grid", 
                           sort_by = "accuracy", 
                           decreasing = TRUE)
print(dl_gridperf)

best_dl_model_id <- dl_gridperf@model_ids[[1]]
best_dl <- h2o.getModel(best_dl_model_id)
h2o.performance(best_dl, test)







