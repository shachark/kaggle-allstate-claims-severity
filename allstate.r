if (.Platform$OS.type == 'windows') {
  Sys.setenv(JAVA_HOME = 'D:\\Program Files\\Java\\jdk1.7.0_79')
}
options(java.parameters = '-Xmx28g')

library(needs)
needs(skaggle, data.table, xgboost, Hmisc, Matrix, entropy, RANN, glmnet, corrplot, nnls, statmod, mxnet, extraTrees, gbm)

# Configuration
# ==================================================================================================

config = create.config('MAE', mode = 'cv', layer = 2)

config$do.base.preprocess = F
config$do.preprocess      = F
config$do.experiment      = F
config$do.train           = F
config$do.submit          = T

#
# Preprocessing parameters
#

config$id.field = 'id'
config$target.field = 'loss'
config$target.type = 'numeric'

config$nr.folds = 8
config$nr.partitions = 1
config$data.seed = 123

config$add.summary.stats.per.row = T
config$add.autoenc               = F
config$add.num.interactions      = F
config$add.hexv.encoding         = T
config$add.freq.encoding         = F
config$add.yenc                  = T
config$add.loo.knn.features      = F # this leaks badly here
config$add.cv.glmnet.features    = F
config$add.cv.keras.features     = F
config$ohe.categoricals          = F

config$target.transform              = 'log' # { none, sqrt, qrt, log, experiment }
config$scale.all                     = F
config$bin.nr.levels                 = 30
config$nr.cat2.hexv.features         = 500
config$nr.cat2.yenc.features         = 100
config$max.levels.to.sms.bayes       = 30  # FIXME try tuning this
config$sms.bayes.count               = 200
config$max.nr.numerical.sms.levels   = 30  # FIXME try tuning this
config$glmnet.interaction.order      = 2 # 4

if (config$target.transform == 'none') {
  config$tt     = function(y) y
  config$tt.inv = function(y) y
} else if (config$target.transform == 'sqrt') {
  config$tt     = function(y) sqrt(y)
  config$tt.inv = function(y) y ^ 2
} else if (config$target.transform == 'qrt') {
  config$tt     = function(y) y ^ 0.25
  config$tt.inv = function(y) y ^ 4
} else if (config$target.transform == 'log') {
  config$tt     = function(y) log(y + 200)
  config$tt.inv = function(y) exp(y) - 200
} else if (config$target.transform == 'experiment') {
  # An inverse Gaussian distribution with mean = 1.045931, shape = 1.096265 (when multiplied by
  # 2904.086) fits the marginal label distribution pretty well.
  # There are what seems to be a few (100?) outliers captured by a QQ plot of the MLE fit and the 
  # empirical quantiles. Maybe its best to exclude these or develop a separate model for them?
  # Hmm.. well the implementation of qinvgauss in statmod can't handle probabilities too close to 
  # 0 or 1, so I can't easily try this idea.
  #config$tt     = function(y) qnorm(pinvgauss(y / 2903.086, mean = 1.045931, shape = 1.096265))
  #config$tt.inv = function(y) qinvgauss(pnorm(y), mean = 1.045931, shape = 1.096265) * 2903.086
  #
  # So instead let's try with a worse but maybe still useful gamma fit.
  config$tt     = function(y) qnorm(pgamma(y / 2903.086, shape = 1.803671, rate = 1.492131))
  config$tt.inv = function(y) qgamma(pnorm(y), shape = 1.803671, rate = 1.492131) * 2903.086
} else {
  stop('wtf')
}

config$eval.score = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  err = mean(abs(config$tt.inv(labels) - config$tt.inv(preds)))
  return (list(metric = 'mae', value = err))
}
config$eval.score.ltd = function(preds, dtrain, ntreelimits) {
  labels = config$tt.inv(getinfo(dtrain, 'label'))
  mae = NULL
  for (i in 1:ncol(preds)) {
    mae = c(mae, mean(abs(labels - config$tt.inv(preds[, i]))))
  }
  names(mae) = ntreelimits
  return (list(metric = 'mae', value = mae))
}
config$eval.score.core = function(preds, labels) {
  mean(abs(config$tt.inv(labels) - config$tt.inv(preds)))
}
config$eval.score.core.multi = function(preds, labels) {
  mae = NULL
  labels = config$tt.inv(labels)
  for (i in 1:ncol(preds)) {
    mae = c(mae, mean(abs(labels - config$tt.inv(preds[, i]))))
  }
  return (mae)
}
config$score.to.maximize = F

# I gotta admit that I'm not clear on why this makes sense
config$fair = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  x = preds - labels
  z = 2 / (abs(x) + 2)
  grad = x * z
  hess = z ^ 2
  return (list(grad = grad, hess = hess))
}

#
# Training parameters
#

if (config$layer == 0) {
  config$model.tag = 'new.xgb1'
  config$model.select = 'xgb'
  config$rng.seed = 1234
  
  config$xgb.params = list(
   #objective           = 'reg:linear', 
   #objective           = 'reg:gamma', 
    objective           = config$fair,
    booster             = 'gbtree',
   #booster             = 'gblinear',
    eval_metric         = config$eval.score,
    maximize            = F,
    nrounds             = 6000,
    eta                 = 0.01,
    max_depth           = 20,
   #subsample           = 0.7,
    colsample_bytree    = 0.1,
    gamma               = 1.8,
   #alpha               = 6,
   #lambda              = 10,
    min_child_weight    = 75,
    base_score          = 7.75,
   #num_parallel_tree   = 5,
    annoying = T
  )

  config$xgb.ntreelimits.every = 300
  #config$xgb.early.stop.round = 300
  config$xgb.params$print.every.n = 100
  
  config$h2o.gbm.params = list(
    distribution             = 'laplace',
    ntrees                   = 3000,
    learn_rate               = 0.02, 
    max_depth                = 8, 
    sample_rate              = 0.9,
    col_sample_rate_per_tree = 0.3,
    min_split_improvement    = 10,
    min_rows                 = 300,
    annoying = T
  )
  
  config$gbm.params = list(
    distribution      = 'laplace', # gaussian, laplace, tdist
    n.trees           = 3000,
    pred.n.trees      = seq(100, 3000, by = 100),
    shrinkage         = 0.02,
    interaction.depth = 8,
    bag.fraction      = 0.9,
    n.minobsinnode    = 300,
    all.var.monotone  = F
  )
  
  config$ranger.params = list(
    num.trees = 1000,
    mtry = 30
  )
  
  config$rf.params = list(
    ntree = 1000,
    mtry = 30,
    nodesize = 50, # only for randomForest
    max.depth = 50, # only for h2o
    binomial_double_trees = F
  )
  
  config$et.params = list(
    ntree = 200,
    mtry = 30,
    nodesize = 20
  )
  
  config$glmnet.params = list(
    family = 'gaussian',
    lambda = 10 ^ seq(-2, -5, len = 10),
    alpha = 1 # i.e., only L1 regularization
  )
  
  config$knn.params = list(
    k = 51,
    eps = 0 # even 1 is much faster
  )
  
  config$mxnet.params = list(
    define.net = function(config) {
      inp  = mx.symbol.Variable('data')
      l1   = mx.symbol.FullyConnected(inp, name = 'l1', num.hidden = 400)
      a1   = mx.symbol.Activation(l1, name = 'a1', act_type = 'relu')
      d1   = mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
      l2   = mx.symbol.FullyConnected(d1, name = 'l2', num.hidden = 200)
      a2   = mx.symbol.Activation(l2, name = 'a2', act_type = 'relu')
      d2   = mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
      l3   = mx.symbol.FullyConnected(d2, name = 'l3', num.hidden = 50)
      a3   = mx.symbol.Activation(l3, name = 'a3', act_type = 'relu')
      d3   = mx.symbol.Dropout(a3, name = 'd3', p = 0.2)
      l4   = mx.symbol.FullyConnected(d3, name = 'l4', num.hidden = 1)
      outp = mx.symbol.LinearRegressionOutput(l4, name = 'outp')
      #outp = mx.symbol.MAERegressionOutput(l4, name = 'outp')

      return (outp)
    },

    optimizer = 'sgd', # options 12/2016 {sgd, rmsprop, adam, adagrad, adadelta}

    sgd.learning.rate = 0.01,
    sgd.momentum      = 0.3,
    sgd.wd            = 0,
    
    batch.size    = 100,
    num.round     = 20, # aka "epoch"
    nr.seed.reps  = 1, # apparently 5 to 10 of these reduce the error by about 3
    annoying = T
  )

  if (config$target.transform == 'none') {
    config$mxnet.params$mx.metric = mx.metric.mae
  } else if (config$target.transform == 'sqrt') {
    config$mxnet.params$mx.metric = mx.metric.custom('mae', function(label, pred) {
      return (mean(abs(label ^ 2 - pred ^ 2)))
    })
  } else if (config$target.transform == 'qrt') {
    config$mxnet.params$mx.metric = mx.metric.custom('mae', function(label, pred) {
      return (mean(abs(label ^ 4 - pred ^ 4)))
    })
  } else if (config$target.transform == 'log') {
    config$mxnet.params$mx.metric = mx.metric.custom('mae', function(label, pred) {
      return (mean(abs(exp(label) - exp(pred))))
    })
  } else {
    stop('wtf')
  }
} else if (config$layer == 1) {
  config$model.tag = 'stack_l1_xgb'
  config$model.select = 'xgb'
  config$rng.seed = 9999
  config$stack.tags  = c('xgb0', 'xgb1', 'xgb2', 'xgb3', 'xgb4', 'xgb5', 'glmnet0', 'glmnet1', 'gbm0', 'ranger0', 'et0', 'knn0', 'mxnet0', 'new.xgb0', 'new.xgb1', 'new.xgb2')
  config$stack.trans = c('log' , 'sqrt', 'log' , 'log' , 'log' , 'qrt' , 'log'    , 'log'    , 'none', 'log'    , 'log', 'log' , 'log'   , 'log'     , 'log'     , 'log'     ) # FIXME ok I should have done this before saving the predictions to disk...
  #config$stack.tags  = c('xgb0', 'xgb2', 'xgb3', 'xgb4', 'xgb5', 'gbm0', 'new.xgb1', 'new.xgb2')
  #config$stack.trans = c('log' , 'log' , 'log' , 'log' , 'qrt' , 'none', 'log'     , 'log'     ) # FIXME ok I should have done this before saving the predictions to disk...
  config$stack.keras = T
  
  config$belnd.fun = median
  
  config$xgb.params = list(
    objective           = 'reg:linear', 
    #objective           = 'reg:gamma', 
    #objective           = config$fair,
    booster             = 'gbtree',
    #booster             = 'gblinear',
    eval_metric         = config$eval.score,
    maximize            = F,
    nrounds             = 5000,
    eta                 = 0.01,
    max_depth           = 3,
    subsample           = 0.95,
    colsample_bytree    = 0.5,
    gamma               = 0,
    #alpha               = 6,
    #lambda              = 10,
    min_child_weight    = 20,
    #base_score          = 7.75,
    #num_parallel_tree   = 5,
    annoying = T
  )
  
  config$gbm.params = list(
    distribution      = 'laplace', # gaussian, laplace, tdist
    n.trees           = 3000,
    pred.n.trees      = seq(100, 3000, by = 100),
    shrinkage         = 0.02,
    interaction.depth = 8,
    bag.fraction      = 0.9,
    n.minobsinnode    = 300,
    all.var.monotone  = T
  )
  
  config$h2o.gbm.params = list(
    distribution             = 'laplace',
    ntrees                   = 5000,
    learn_rate               = 0.01, 
    max_depth                = 3, 
    sample_rate              = 0.95,
    col_sample_rate_per_tree = 0.5,
    min_rows                 = 20,
    min_split_improvement    = 0,
    annoying = T
  )
  
  config$glmnet.params = list(
    family = 'gaussian',
    lambda = 10 ^ seq(-2, -5, len = 10),
    alpha = 0.1 # 0 means only L2 regularization, 1 means only L1 regularization
  )
  
  config$nnls.normalize = F
  config$glm.family = 'gaussian'
} else if (config$layer == 2) {
  config$model.tag = 'stack_l2'
  config$model.select = 'custom'
  config$rng.seed = 9999
  config$stack.tags  = c('stack_l1_mlae', 'stack_l1_h2gbm_2', 'stack_l1_xgb')
  config$stack.trans = c('none'         , 'none'            , 'log'         ) # FIXME ok I should have done this before saving the predictions to disk...
  config$stack.keras = F
  
  config$belnd.fun = median
  
  config$xgb.params = list(
    #objective           = 'reg:linear', 
    #objective           = 'reg:gamma', 
    objective           = config$fair,
    booster             = 'gbtree',
    #booster             = 'gblinear',
    eval_metric         = config$eval.score,
    maximize            = F,
    nrounds             = 5000,
    eta                 = 0.01,
    max_depth           = 8,
    subsample           = 0.7,
    colsample_bytree    = 0.9,
    gamma               = 2.5,
    #alpha               = 6,
    #lambda              = 10,
    min_child_weight    = 80,
    #base_score          = 7.75,
    #num_parallel_tree   = 5,
    annoying = T
  )
  
  config$gbm.params = list(
    distribution      = 'laplace', # gaussian, laplace, tdist
    n.trees           = 3000,
    pred.n.trees      = seq(100, 3000, by = 100),
    shrinkage         = 0.02,
    interaction.depth = 8,
    bag.fraction      = 0.9,
    n.minobsinnode    = 300,
    all.var.monotone  = T
  )
  
  config$h2o.gbm.params = list(
    distribution             = 'laplace',
    ntrees                   = 5000,
    learn_rate               = 0.01, 
    max_depth                = 3, 
    sample_rate              = 0.95,
    col_sample_rate_per_tree = 0.5,
    min_rows                 = 20,
    min_split_improvement    = 0,
    annoying = T
  )
  
  config$glmnet.params = list(
    family = 'gaussian',
    lambda = 10 ^ seq(-2, -5, len = 10),
    alpha = 0.1 # 0 means only L2 regularization, 1 means only L1 regularization
  )
  
  config$nnls.normalize = F
  config$glm.family = 'gaussian'
} else {
  stop('wtf')
}

# Misc.
config$measure.importance = F
config$dat.filename = paste0(config$tmp.dir, '/base-data.RData')

#
# Submission parameters
#

config$submt.id = 7
config$submt.column = 1
config$ref.submt.id = '345-blend' # PubLB 1102.6

# ==================================================================================================
# Preprocessing

config$preprocess.raw1 = function(config) {
  cat(date(), 'Preprocessing (phase1)\n') # (this part is supposed to be pretty much the same for all models)
  
  cat(date(), 'Loading raw data\n')
  train = fread(paste0(config$data.dir, '/train.csv'), stringsAsFactors = T)
  test  = fread(paste0(config$data.dir, '/test.csv') , stringsAsFactors = T)

  train.ids = train[[config$id.field]]
  test.ids  = test [[config$id.field]]
  train.labels = train[[config$target.field]]
  dat = rbind(train[, setdiff(names(train), config$target.field), with = F], test)
  
  train.idx = 1:length(train.labels)
  test.idx = (length(train.labels) + 1):nrow(dat)
  rm(train, test)
  gc()
  
  # Analyze the raw features
  cat(date(), 'Raw data contains', ncol(dat), 'features\n')
  original.feature.names = copy(names(dat))
  
  cat.features = names(which(sapply(dat, is.factor)))
  cat.vars.and.levels = lapply(dat[, cat.features, with = F], levels)
  
  review.features = function(dat) {
    as.data.table(data.frame(
      type      = unlist(sapply(dat, function(x) class(x)[1])),
      n.unique  = unlist(sapply(dat, function(x) length(unique(x)))),
      f.missing = unlist(sapply(dat, function(x) mean(is.na(x)))),
      spear.cor = unlist(sapply(dat[train.idx], function(x) { idx = !is.na(x); if (!(class(x)[1] %in% c('integer', 'numeric'))) return (NA); cor(x[idx], y = train.labels[idx], method = 'spearman') }))
    ), keep.rownames = T)[order(abs(spear.cor), decreasing = T)]
  }

  raw.features.analysis = review.features(dat)
  if (0) {  
    cat(date(), 'Raw features:\n\n')
    save(raw.features.analysis, file = 'raw-feature-analysis.RData')
    print(raw.features.analysis)
  }

  #
  # CV folds that accompany the data throughout the training stages
  #

  set.seed(config$data.seed)
  
  if (0) {
    cat(date(), 'NOTE: assuming random CV partitioning is appropriate (mimics the train/test split for example)\n')
    cv.folds = NULL
    for (i in 1:config$nr.partitions) {
      cv.folds0 = createFolds(train.labels, k = config$nr.folds / config$nr.partitions)
      names(cv.folds0) = paste0(names(cv.folds0), '.', i)
      cv.folds = c(cv.folds, cv.folds0)
    }
  } else {
    # FIXME what makes for a good binning? I think I want to make sure the highest values are spread evenly
    # What I'm doing here is not random (on top of the original sampling). Not sure if this will create a
    # problematic difference between folds.
    stopifnot(config$nr.partitions == 1)
    cat(date(), 'NOTE: using a weird nonrandom stratified CV partitioning\n')
    fold.idx = rep(c(1:config$nr.folds, config$nr.folds:1), ceiling(length(train.labels) / config$nr.folds))[rank(train.labels, ties.method = 'random')]
    cv.folds = list()
    for (i in 1:config$nr.folds) {
      cv.folds[[paste0('Fold', i)]] = which(fold.idx == i)
    }
  }
  
  set.seed(config$data.seed)
  
  # The formus seem to suggest there is no leak
  dat[, (config$id.field) := NULL]
  
  # Let's treat all the binary features as numeric right off the bat
  cols = raw.features.analysis[type == 'factor' & n.unique == 2, rn]
  for (j in cols) set(dat, j = j, value = as.integer(dat[[j]]) - 1)
  
  # And lexicographically (hexavegisimal) reorder the levels of the other categoricals
  reorder.levels.hexv = function (x) {
    lvls = as.character(unique(x))
    lvls = lvls[order(nchar(lvls), tolower(lvls))]
    factor(x, levels = lvls)
  }
  cols = raw.features.analysis[type == 'factor' & n.unique > 2, rn]
  for (j in cols) set(dat, j = j, value = reorder.levels.hexv(dat[[j]]))
  
  # <----- FIXME place for experiments. After debugging, will be moved
  #
  if (config$add.num.interactions) {
    cat(date(), 'Adding numerical interactions\n')
    cols = intersect(names(dat)[which(sapply(dat, function(x) !is.factor(x) & uniqueN(x) > 3))], original.feature.names)
    pp = preProcess(dat[, cols, with = F], method = 'YeoJohnson')
    dat = cbind(dat, model.matrix(~ . ^ 3 - . - 1, as.data.frame(predict(pp, dat[, cols, with = F]))))
  }
  #
  # ----->

  ##################################################################################################
  # Dimensionality reduction
  ##################################################################################################
  
  if (config$add.summary.stats.per.row) {
    cat(date(), 'Adding summary stats per row\n')
    
    num.features = sapply(dat, function(x) uniqueN(x) > 2 & class(x)[1] %in% c('integer', 'numeric'))
    numdat = data.matrix(dat[, num.features, with = F])
    
    if (0) {
      # Simple summary statistics of numericals per row (doesn't seem to help)
    dat[, summary.maxnum := apply(numdat, 1, max   )]
    dat[, summary.minnum := apply(numdat, 1, min   )]
    dat[, summary.mednum := apply(numdat, 1, median)]
    dat[, summary.stdnum := apply(numdat, 1, sd    )]
    }
    
    # Simple summary statistics of numericals per row - on scaled features
    numdat = scale(numdat)
    dat[, summary.maxnum.s := apply(numdat, 1, max   )]
    dat[, summary.minnum.s := apply(numdat, 1, min   )]
    dat[, summary.mednum.s := apply(numdat, 1, median)]
    
    if (0) {
      # Count quantile-binned numericals per row (doesn't seem to help)
    qtls = t(as.data.frame(apply(numdat, 2, quantile, probs = c(0.1, 0.25, 0.5, 0.75, 0.9))))
    numdat = t(numdat)
    dat[, qcnt.numerics1 := colSums(numdat <= qtls[, 1])]
    dat[, qcnt.numerics2 := colSums(numdat > qtls[, 1] & numdat <= qtls[, 2])]
    dat[, qcnt.numerics3 := colSums(numdat > qtls[, 2] & numdat <= qtls[, 3])]
    dat[, qcnt.numerics4 := colSums(numdat > qtls[, 3] & numdat <= qtls[, 4])]
    dat[, qcnt.numerics5 := colSums(numdat > qtls[, 4] & numdat <= qtls[, 5])]
    dat[, qcnt.numerics6 := colSums(numdat > qtls[, 5])]
    }
    
    rm(numdat)
    gc()
  }
  
  #
  # Autoencoder features
  #
  
  if (config$add.autoenc) {
    cat(date(), 'Adding autoencoder 2d mapping\n')
    stop('TODO')
    load(file = 'autoencoder-output.RData') # => deep.fea (generated in explore.r, it takes a long time)
    dat = cbind(dat, dr.autoenc = deep.fea)
    rm(deep.fea)
  }

  ##################################################################################################
  # Simple numerical encodings of categoricals (and optionally of binned numericals as categoricals)
  ##################################################################################################
  
  #
  # Recode categoricals assuming the (letter based) levels are meaningful somehow
  #
  
  if (config$add.hexv.encoding) {
    cat(date(), 'Adding hexv encoded categoricals\n')
    cols = intersect(names(dat)[sapply(dat, function(x) class(x)[1] == 'factor')], original.feature.names)
    if (truelength(dat) < ncol(dat) + length(cols)) alloc.col(dat, ncol(dat) + length(cols))
    for (col in cols) set(dat, j = paste0('hexv.', col), value = as.integer(dat[[col]]) - 1) # since levels are already sorted in hexv order

    num.features = intersect(setdiff(names(dat)[which(sapply(dat, function(x) !is.factor(x) & uniqueN(x) > 3))], config$id.field), original.feature.names)
    new.ncol = ncol(dat) + length(num.features) 
    if (truelength(dat) < new.ncol) alloc.col(dat, new.ncol)
    for (j in num.features) set(dat, j = paste0('tmp.', j), value = cut2(dat[[j]], g = min(config$bin.nr.levels, uniqueN(dat[[j]]))))
    
    load('assoc-ord2.RData') # => ord2
    ord2 = head(ord2[order(d12, decreasing = T)], config$nr.cat2.hexv.features)
    ord2[, j1 := sub('^cont', 'tmp.cont', x1)]
    ord2[, j2 := sub('^cont', 'tmp.cont', x2)]
    new.ncol = ncol(dat) + config$nr.cat2.hexv.features 
    if (truelength(dat) < new.ncol) alloc.col(dat, new.ncol)
    for (j in 1:nrow(ord2)) {
      xx = interaction(dat[[ord2$j1[j]]], dat[[ord2$j2[j]]], drop = T, lex.order = ord2$lex.order[j])
      set(dat, j = paste0('hexv.', ord2$x1[j], 'X', ord2$x2[j]), value = as.integer(xx) - 1)
    }
    
    dat[, (paste0('tmp.', num.features)) := NULL]
  }
  
  #
  # Frequency encoding
  #
  
  if (config$add.freq.encoding) {
    cat(date(), 'Adding frequency encoded categoricals\n')
    cols = intersect(names(dat)[sapply(dat, function(x) between(uniqueN(x), 3, 512))], original.feature.names)
    if (truelength(dat) < ncol(dat) + length(cols)) alloc.col(dat, ncol(dat) + length(cols))
    for (col in cols) set(dat, j = paste0('frq.', col), value = rank(-freq.encode(as.factor(dat[[col]]))) - 1)
  }
  
  ##################################################################################################
  # Basal stacking: add features generated by simple k-fold CV like partitioning and train/predict
  ##################################################################################################
  
  #
  # Marginal (regulaized LM / LMM)
  #
  
  if (config$add.yenc) {
    # Ok so these are prone to the leakage-in-stacking issue. I already know that I can still 
    # estimate prediction performance by doing this stacking fully nested within the evaluation CV.
    # This allows more informed modeling desicions like guiding feature engineering and model 
    # selection, but it's very resource consuming (and I'm not sure an accurate performance 
    # estimation is too important given that I don't have a lot of time with this comp). What's 
    # worse is that in any case this doesn't help the performance of the model that suffers from
    # leakage. The leak has to be plugged at the source.
    #
    # To that end, I've got a new idea I could try: generate a leakage-only reference meta-feature
    # of some sort (or multiple such references) and regress (in some way...) the meta-features of
    # interest against this reference, and take the residual (whatever that may be). The idea is 
    # that the residual would have much less leakage in it than the "useful+leak" meta-feature.
    #
    # But for now I'm just going to do it the usual (leakage prone) way.

    # TODO: Need to use GLMM automatically when the cardinality gets too high! (especially relevant 
    # with interactions)
    
    cat.features = intersect(names(dat)[which(sapply(dat, is.factor))], original.feature.names)
    num.features = intersect(setdiff(names(dat)[which(sapply(dat, function(x) !is.factor(x) & uniqueN(x) > 3))], config$id.field), original.feature.names)

    # Allocate new columns for meta features based on each categorical and binned numerical, interactions
    new.ncol = ncol(dat) + length(c(num.features, cat.features)) + config$nr.cat2.yenc.features
    if (truelength(dat) < new.ncol) alloc.col(dat, new.ncol)
    
    # Bin numericals
    for (j in num.features) set(dat, j = paste0('sms.', j), value = cut2(dat[[j]], g = min(config$bin.nr.levels, uniqueN(dat[[j]]))))

    # Stack univariate models:
    y.train = log(train.labels)
    y = c(y.train, rep(NA, length(test.ids)))
    if (0) {
      # naive version
      yenc = function(x) {
        sms.encode(x, y, cv.folds)
      }
    } else {
      # "smart" version
      yenc = function(x) {
        xe = rep(NA_real_, length(x))
        xe[train.idx] = yenc.automatic(x[train.idx], y.train, cv.folds,  0, NULL       , config$max.nr.numerical.sms.levels, config$max.levels.to.sms.bayes, config$sms.bayes.count)
        xe[test.idx ] = yenc.automatic(x[train.idx], y.train, cv.folds, -1, x[test.idx], config$max.nr.numerical.sms.levels, config$max.levels.to.sms.bayes, config$sms.bayes.count)
        return (xe)
      }
    }
    
    # Bivariate (doing this first since the univariate will overwrite the binned numericals)
    load('assoc-cat2.RData') # => cat2
    cat2 = head(cat2[order(p12)], config$nr.cat2.yenc.features)
    cat2[, j1 := sub('^cont', 'sms.cont', x1)]
    cat2[, j2 := sub('^cont', 'sms.cont', x2)]
    for (j in 1:nrow(cat2)) {
      # FIXME this is very slow - need to parallelize it! and/or limit the number of categories to the biggest K?
      xx = interaction(dat[[cat2$j1[j]]], dat[[cat2$j2[j]]], drop = T)
      cat(date(), ' Yenc cat2 ', j, ' of ', nrow(cat2), ' (', cat2$x1[j], ', ', cat2$x2[j], ' => ', uniqueN(xx), ' levels)\n', sep = '')
      set(dat, j = paste0('sms.', cat2$x1[j], 'X', cat2$x2[j]), value = yenc(xx))
    }
    
    # Univariate    
    for (j in paste0('sms.', num.features)) set(dat, j =                j , value = sms.encode(dat[[j]], y, cv.folds))
    for (j in        cat.features         ) set(dat, j = paste0('sms.', j), value = sms.encode(dat[[j]], y, cv.folds))
    
    gc()
  }

  #
  # KNN
  # TODO: this uses LOO. Check this doesn't leak; if it does then switch to CV/NCV
  #
  
  if (config$add.loo.knn.features) {
    cat(date(), 'Generating LOO NN features\n')
    
    dat.nn = copy(dat[, intersect(names(dat), original.feature.names), with = F])
    num.features = intersect(setdiff(names(dat)[which(sapply(dat, function(x) !is.factor(x) & uniqueN(x) > 3))], config$id.field), original.feature.names)
    for (j in num.features) set(dat.nn, j = j, value = cut2(dat.nn[[j]], g = min(config$bin.nr.levels, uniqueN(dat.nn[[j]]))))
    dat.nn = model.matrix(~ . - 1, dat.nn) # FIXME do I need a full rank representation? does it really matter? it's too bad that there is no knn implementation that works on sparse matrices
    y.nn = c(log(train.labels), rep(NA, length(test.idx)))
    
    generate.knn.bf = function(eps.search, k.search, prefix) {
      res = list()
      
      idx1 = c(rep(T, length(train.idx)), rep(F, length(test.idx)))
      nn.res = nn2(dat.nn[idx1, ], query = dat.nn, k = k.search + 1, eps = eps.search)
      res$ypred = rowMeans(matrix(y.nn[nn.res$nn.idx], nrow = nrow(dat.nn)))
      res$nn.dist.1 = ifelse(idx1, nn.res$nn.dists[, 2], nn.res$nn.dists[, 1])
      res$nn.dist.2 = ifelse(idx1, nn.res$nn.dists[, 3], nn.res$nn.dists[, 2])
      res$nn.dist.3 = ifelse(idx1, nn.res$nn.dists[, 4], nn.res$nn.dists[, 3])
      res$nn.dist.k = ifelse(idx1, nn.res$nn.dists[, k.search + 1], nn.res$nn.dists[, k.search])
      res$nn.dist.m = ifelse(idx1, apply(nn.res$nn.dists[, -1], 1, median), apply(nn.res$nn.dists[, -(k.search + 1)], 1, median))
      res$nn.dist.s = ifelse(idx1, apply(nn.res$nn.dists[, -1], 1, sd    ), apply(nn.res$nn.dists[, -(k.search + 1)], 1, sd    ))
      
      res = as.data.frame(res)
      names(res) = paste0(prefix, names(res))
      
      return (res)
    }
    
    dat = cbind(dat, generate.knn.bf(eps.search = 5, k.search = 50, prefix = 'looknn.'))
    rm(dat.nn, y.nn)
  }

  #
  # GLMnet
  # TODO: this uses simple CV. Check this doesn't leak; if it does then switch to CV/NCV
  #
  
  if (config$add.cv.glmnet.features) {
    cat(date(), 'Adding CV GLMnet features\n')
    
    generate.glmnet.mf = function(x, y, xnew, ord) {
      frmla = as.formula(paste0('~.^', ord, '-1'))
      mdl = glmnet(model.matrix(frmla, x), y, family = 'gaussian')
      preds = predict(mdl, model.matrix(frmla, xnew), s = 0, type = 'response') # the best was lm...
    }
    
    generate.lm.mf = function(x, y, xnew, ord) {
      frmla = as.formula(paste0('~.^', ord, '-1'))
      mdl = coef(lm.fit(model.matrix(frmla, x), y))
      preds = c(model.matrix(frmla, xnew) %*% mdl)
    }
    
    #browser()
    ##### Hmm... so far I've seen that as I add higher and higher orders of interaction in the 
    # way of a formulae (y ~ .^j) with j up to 4 (about 1500 features) the optimal LASSO in terms of
    # CV error is with no regularization at all, i.e., an OLS! I wonder how high I can go.
    #cvg = cv.glmnet(model.matrix(~ (.)^3 - 1, train.glmnet), train.labels, nfolds = 5, type.measure = 'mae') # log(train.labels + 200)
    #plot(cvg)
    # FIXME explore this further, build a more elaborate linear model that comes closer to exploiting the feature set
    #####
    
    glmnet.features = intersect(names(dat)[which(sapply(dat, function(x) !is.factor(x) & uniqueN(x) > 3))], original.feature.names)
    pp = preProcess(dat[, glmnet.features, with = F], method = 'YeoJohnson') # FIXME check if this is a good choice here
    dat.glmnet = as.data.frame(predict(pp, dat[, glmnet.features, with = F])) # glmnet will standardize
    dat.glmnet = dat.glmnet[, which(!unlist(lapply(dat.glmnet, function(x) any(is.na(x)))))] # if there are any Infs or constants
    train.glmnet = dat.glmnet[train.idx, ]
    test.glmnet  = dat.glmnet[test.idx , ]
    rm(dat.glmnet)
    gc()
    
    dat[test.idx , stacked.glmnet.num := smm.generic2(train.glmnet, log(train.labels), generate.lm.mf, cv.folds, -1, test.glmnet, ord = config$glmnet.interaction.order)]
    dat[train.idx, stacked.glmnet.num := smm.generic2(train.glmnet, log(train.labels), generate.lm.mf, cv.folds,  0, NULL       , ord = config$glmnet.interaction.order)]
    
    # TODO do I want to try this also on some representation of categoricals?
  }
  
  if (config$add.cv.keras.features) {
    cat(date(), 'Adding CV Keras features\n')
    # NOTE: this used a different CV partition to generate the "OOB" preds. I don't think it's 
    # critical, but it might increase the leak-in-stacking...
    tmp = rbind(fread('keras-kernel-oob-preds.csv')[order(id)], fread('keras-kernel-test-preds.csv')[order(id)])
    dat[, stacked.keras := tmp$loss]
    rm(tmp)
  }
  
  #
  # One hot encoding (leaving this to the end since it means we have to switch to a sparse matrix data structure)
  #
  
  if (config$ohe.categoricals) {
    factor.cols = names(dat)[sapply(dat, is.factor)]
    nonfactor.cols = names(dat)[sapply(dat, function(x) !is.factor(x))]
    
    ridx = as.integer(1:nrow(dat))
    dat.old = dat
    dat = Matrix(data.matrix(dat.old[, nonfactor.cols, with = F]), sparse = T)
    
    gc()
    for (col in factor.cols) {
      #cat(date(), 'adding', col, '\n')
      x = as.integer(dat.old[[col]])
      x[is.na(x)] = max(x, na.rm = T) + 1
      tmp = sparseMatrix(ridx, x)
      colnames(tmp) = paste0(col, '.', 1:ncol(tmp))
      dat = cbind(dat, tmp)
      gc()
    }
    
    rm(dat.old)
    gc()
  } else {
    nonfactor.cols = names(dat)[sapply(dat, function(x) !is.factor(x))]
    dat = Matrix(data.matrix(dat[, nonfactor.cols, with = F]), sparse = T)
  }

  #
  # Save data and ancillary information
  #
  
  cat(date(), 'Saving to disk\n')
  
  ancillary = list()
  ancillary$train.labels = train.labels
  ancillary$train.ids = train.ids
  ancillary$test.ids = test.ids
  ancillary$cv.folds = cv.folds
  ancillary$feature.names = colnames(dat)
  
  save(ancillary, dat, file = config$dat.filename, compress = F)
}

config$preprocess.raw2 = function(config) {
  cat(date(), 'Preprocessing (phase2)\n') # (this is the model-dependent part)

  load(config$dat.filename) # => ancillary, dat
  cat(date(), 'Preprocessed data contains', ncol(dat), 'features\n')

  # Target transform
  ancillary$orig.train.labels = ancillary$train.labels
  ancillary$train.labels = config$tt(ancillary$train.labels)

  #
  # Impute and scale
  #
  
  # Most models can't handle NAs, and those that can are (usually) better off without this  
  if (!config$model.select %in% c('xgb', 'gbm', 'rf', 'nb')) {
    cat(date(), 'Imputing all missing values\n')
    dat = as.matrix(dat)
    dat = randomForest::na.roughfix(dat)
    still.na = (colSums(is.na(dat)) > 0)
    if (any(still.na)) {
      cat('Some NAs still in', colnames(dat)[still.na], '? dropping\n')
      dat = dat[, !still.na]
    }
    dat = as.data.table(dat)
  }
  
  # Some models work better on scaled features (e.g., nnet, maybe knn)
  if (config$scale.all) {
    cat(date(), 'Scaling all features\n')
    dat = as.data.table(scale(dat))
  }

  #
  # save to disk
  #
  
  save(ancillary, file = config$ancillary.filename)
  
  train = dat[1:length(ancillary$train.labels), ]
  test  = dat[(length(ancillary$train.labels) + 1):nrow(dat) , ]
  rm(dat)
  gc()
  
  if (config$model.select == 'xgb') {
    if (class(train) == 'dgCMatrix') {
      # EXPERIMENT: does xgb.DMatrix treat zeros as missing when given a dgCMatrix input?!
      cat('EXPERIMENT dense\n')
      train = as.matrix(train)
      test  = as.matrix(test )
    }
    xgb.DMatrix.save(xgb.DMatrix(train, label = ancillary$train.labels), config$xgb.trainset.filename)
    xgb.DMatrix.save(xgb.DMatrix(test                                 ), config$xgb.testset.filename )
  } else if (config$model.select == 'h2o.gbm') {
    # Well, this doesn't work :(
    #write.svmlight(train, ancillary$train.labels, config$xgb.trainset.filename)
    #write.svmlight(test , rep(0, nrow(test))    , config$xgb.testset.filename )
    train = as.data.frame(as.matrix(train))
    test  = as.data.frame(as.matrix(test ))
    train$target = ancillary$train.labels
    save(train, test, file = config$dataset.filename, compress = F)
  } else {
    # will need a lot of ram for this..
    train = as.data.frame(as.matrix(train))
    test  = as.data.frame(as.matrix(test ))
    train$target = ancillary$train.labels
    save(train, test, file = config$dataset.filename, compress = F)
  }
}

config$preprocess.stack = function(config) {
  load(paste0(config$tmp.dir, '/pp-data-ancillary-L', config$layer - 1, '.RData')) # => ancillary

  # Target transform
  ancillary$train.labels = config$tt(ancillary$orig.train.labels)

  train = NULL
  test = NULL
  
  #stack.dir = config$tmp.dir
  stack.dir = 'tmp'
  
  select.best.tuning = T
  
  cat(date(), 'Loading meta-features\n')
  
  for (imdl in seq_along(config$stack.tags)) {
    model.tag = config$stack.tags[imdl]
    cat('Including model', model.tag)
    
    load(paste0(stack.dir, '/test-preds-', model.tag, '.RData')) # => preds
    if (config$stack.trans[imdl] == 'none') {
      # nothing
    } else if (config$stack.trans[imdl] == 'sqrt') {
      preds = preds ^ 2
    } else if (config$stack.trans[imdl] == 'qrt') {
      preds = preds ^ 4
    } else if (config$stack.trans[imdl] == 'log') {
      preds = exp(preds) - 200
    } else {
      stop('wtf')
    }
    model.preds = data.frame(config$tt(preds))
    if (ncol(model.preds) == 1) {
      names(model.preds) = model.tag
      cat(' => 1 model')
    } else {
      names(model.preds) = paste(model.tag, 1:ncol(model.preds), sep = '_')
      cat(' =>', ncol(model.preds), 'models')
    }
    
    cv.preds = as.data.frame(matrix(NA, length(ancillary$train.labels), ncol(model.preds)))
    names(cv.preds) = names(model.preds)
    for (i in 1:config$nr.folds) {
      load(paste0(stack.dir, '/cv-preds-', model.tag, '-', i, '.RData')) # => preds
      if (config$stack.trans[imdl] == 'none') {
        # nothing
      } else if (config$stack.trans[imdl] == 'sqrt') {
        preds = preds ^ 2
      } else if (config$stack.trans[imdl] == 'qrt') {
        preds = preds ^ 4
      } else if (config$stack.trans[imdl] == 'log') {
        preds = exp(preds) - 200
      } else {
        stop('wtf')
      }
      cv.preds[ancillary$cv.folds[[i]], ] = config$tt(preds)
    }
    
    if (ncol(cv.preds) > 1 && select.best.tuning) {
      cvscores = rep(NA, ncol(cv.preds))
      for (i in 1:ncol(cv.preds)) {
        cvscores[i] = config$eval.score.core(cv.preds[[i]], ancillary$train.labels)
      }
      best.idx = ifelse(config$score.to.maximize, which.max(cvscores), which.min(cvscores))
      model.preds = model.preds[, best.idx, drop = F]
      cv.preds    = cv.preds   [, best.idx, drop = F]
      cat(' => selecting only column ', best.idx, ' (score ', cvscores[best.idx], ')\n', sep = '')
    } else if (select.best.tuning) {
      cvscore = config$eval.score.core(cv.preds[[1]], ancillary$train.labels)
      cat(' (score ', cvscore, ')\n', sep = '')
    } else {
      cat('\n')
    }
    
    if (is.null(test)) {
      test = model.preds
      train = cv.preds
    } else {
      test = cbind(test, model.preds)
      train = cbind(train, cv.preds)
    }
  }
  
  if (config$stack.keras) {
    cat('Including model Keras')
    # NOTE: this used a different CV partition to generate the "OOB" preds. I don't think it's 
    # critical, but it might increase the leak-in-stacking...
    test  = cbind(test , keras = fread('keras-kernel-test-preds.csv')[order(id), config$tt(loss)])
    train = cbind(train, keras = fread('keras-kernel-oob-preds.csv' )[order(id), config$tt(loss)])
    if (select.best.tuning) {
      cvscore = config$eval.score.core(train[[ncol(train)]], ancillary$train.labels)
      cat(' => 1 model (score ', cvscore, ')\n', sep = '')
    } else {
      cat('\n')
    }
  }
  
  if (0) {
    # Look at the CV scores and correlation of the models we are going to stack
    cvscores = data.frame(cv.score = rep(NA, ncol(train)))
    rownames(cvscores) = names(train)
    for (i in 1:ncol(train)) {
      cvscores$cv.score[i] = config$eval.score.core(train[[i]], ancillary$train.labels)
    }
    d = cor(train, method = 'spearman', use = 'complete.obs')
    corrplot(d, is.corr = F, method = 'color', col = colorRampPalette(c('white', 'green', 'red'))(40), addCoef.col = 'black', order = 'AOE', title = 'Score correlations of constituent models')
    #View(cvscores)
    #View(d)
  }
  
  if (0) {
    # Average some models
    train$glmnet01.avg = 0.5 * (train$glmnet0_9 + train$glmnet1_9)
    train$glmnet0_9 = train$glmnet1_9 = NULL
    test$glmnet01.avg = 0.5 * (test$glmnet0_9 + test$glmnet1_9)
    test$glmnet0_9 = test$glmnet1_9 = NULL
    
    train$xgb.avg = 1/7 * (train$xgb0_10 + train$xgb1_10 + train$xgb2_10 + train$xgb3_44 + train$xgb4_50 + train$xgb5_8 + train$gbm0)
    train$xgb0_10 = train$xgb1_10 = train$xgb2_10 = train$xgb3_44 = train$xgb4_50 = train$xgb5_8 = train$gbm0 = NULL
    test$xgb.gbm.avg = 1/7 * (test$xgb0_10 + test$xgb1_10 + test$xgb2_10 + test$xgb3_44 + test$xgb4_50 + test$xgb5_8 + test$gbm0)
    test$xgb0_10 = test$xgb1_10 = test$xgb2_10 = test$xgb3_44 = test$xgb4_50 = test$xgb5_8 = test$gbm0 = NULL
  }

  # Some models work better on scaled features (e.g., nnet, maybe knn)
  if (config$scale.all) {
    cat(date(), 'Scaling all features\n')
    dat = as.data.frame(scale(rbind(train, test)))
    train = dat[1:length(ancillary$train.labels), ]
    test  = dat[(length(ancillary$train.labels) + 1):nrow(dat) , ]
    rm(dat)
    gc()
  }
  
  # Override per-level ancillary info
  ancillary$feature.names = names(train)
  
  cat(date(), 'Saving data\n')
  
  # Data for everything but XGB
  train$target = ancillary$train.labels
  save(ancillary, file = config$ancillary.filename)
  save(train, test, file = config$dataset.filename)
  train$target = NULL
  
  # Data for XGB  
  dtrain = xgb.DMatrix(dat = data.matrix(train), label = ancillary$train.labels)
  dtest  = xgb.DMatrix(dat = data.matrix(test )                                )
  xgb.DMatrix.save(dtrain, config$xgb.trainset.filename)
  xgb.DMatrix.save(dtest , config$xgb.testset.filename)
}

config$preprocess = function(config) {
  if (config$layer == 0) {
    config$preprocess.raw2(config)
    gc()
  } else {
    config$preprocess.stack(config)
    gc()
  }
}

config$finalize.l0.data = function(config) {
}

config$postprocess = function(config, preds) {
  return (config$tt.inv(preds))
}

config$experiment = function(config) {
  dtrain = xgb.DMatrix(config$xgb.trainset.filename)
  load(config$ancillary.filename) # => ancillary
  
  if (1) {
    # Experiment with different target transforms
    
    if (0) { # log with offset (train MAE 1005-)
      setinfo(dtrain, 'label', log(ancillary$train.labels + 200))
      config$eval.score = function(preds, dtrain) {
        labels = getinfo(dtrain, 'label')
        err = mean(abs(exp(labels) - exp(preds)))
        return (list(metric = 'mae', value = err))
      }
    } else if (1) { # sqrt  (804--)
      setinfo(dtrain, 'label', sqrt(ancillary$train.labels))
      config$eval.score = function(preds, dtrain) {
        labels = getinfo(dtrain, 'label')
        err = mean(abs(labels ^ 2 - preds ^ 2))
        return (list(metric = 'mae', value = err))
      }
    }
    
    config$xgb.params = list(
      objective           = 'reg:linear', 
      booster             = 'gbtree',
      eval_metric         = config$eval.score,
      maximize            = F,
      nrounds             = 500,
      eta                 = 0.1,
      max_depth           = 8,
      subsample           = 0.7, #0.8,
      colsample_bytree    = 0.3, #0.5,
      gamma               = 2,
      alpha               = 1,
      annoying = T
    )
    
    xgb.fit = xgb.train(
      params            = config$xgb.params,
      nrounds           = config$xgb.params$nrounds,
      maximize          = F,
      data              = dtrain,
      watchlist         = list(train = dtrain),
      print_every_n     = 20
    )
  }

  if (0) {
    # Run IRLS a couple of times to get weights that I'll later use in a bigger run
    # NOTE - I couldn't get this to work... it's not clear how to do the reweighting
    
    y = getinfo(dtrain, 'label')
    n = length(y)
    w = rep(1, n)
    min.e = 20    # good enough error
    max.e = 15000 # error in outlying cases
  
    if (1) { # for debugging
      config$xgb.params = list(
        objective           = 'reg:linear', 
        booster             = 'gbtree',
        eval_metric         = config$eval.score,
        maximize            = F,
        nrounds             = 500,
        eta                 = 0.1,
        max_depth           = 8,
        subsample           = 0.7, #0.8,
        colsample_bytree    = 0.3, #0.5,
        gamma               = 2,
        alpha               = 1,
        annoying = T
      )
    }
  
    for (i in 1:10) {
      if (i > 1) setinfo(dtrain, 'weight', w)
  
      xgb.fit = xgb.train(
        params            = config$xgb.params,
        nrounds           = config$xgb.params$nrounds,
        maximize          = F,
        data              = dtrain,
        watchlist         = list(train = dtrain),
        print_every_n     = 100
      )
      
      preds = predict(xgb.fit, dtrain)
      mae = config$eval.score(preds, dtrain)$value
      cat(date(), 'MAE', mae, '\n')
      
      e = abs(preds - y)
  
      # FIXME tune this transform to be a gentle nudge in the right direction
      #w = ifelse(e < max.e, pmax(e, min.e) ^ (-0.1), 0)
      w = c(seq(1, 0.9, len = sum(e < max.e)), rep(0, sum(e >= max.e)))[rank(e)]
      
      #browser()
      #plot(e, w, pch = '.')
      #hist(w, 100, main = paste('IRLS iteration', i, 'MAE', round(mae)))
      
      w = w * (n / sum(w)) # seems that XGB expects this to be normalized (I guess some parameters assume this scale to keep making sense)
    }
  }
}

# Non-negative least absolute error
config$train.custom = function(config) {
  load(config$dataset.filename) # => train, test
  load(config$ancillary.filename) # => ancillary
  
  if (!is.null(config$in.fold) && config$in.fold != -1) {
    vidx = ancillary$cv.folds[[config$in.fold]]
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  } else if (config$holdout.validation) {
    set.seed(config$data.seed) # NOTE: A fixed seed to choose the validation set
    vidx = sample(nrow(train), 0.1 * nrow(train))
    valid = train[vidx, ]
    train = train[(1:nrow(train))[-vidx], ]
  }
  
  cat(date(), 'Training NNLAE\n')
  
  set.seed(config$rng.seed)
  y = train$target
  X = data.matrix(train[, -which(names(train) == 'target'), drop = F])
  m = ncol(X)
  
  eval_f = function(pars) {
    yp = c(X  %*% c(pars, 1 - sum(pars)))
    mean(abs(yp - y))
  }

  sol = nloptr::nloptr(x0 = rep(1/m, m - 1), eval_f = eval_f, lb = rep(0, m - 1), ub = rep(1, m - 1),
    opts = list(algorithm = 'NLOPT_LN_COBYLA', ftol_abs = 1e-4, maxeval = 1000))
  
  mdl = c(sol$solution, 1 - sum(sol$solution))
  #mdl[mdl < 1e-8] = 0
  #mdl = mdl / sum(mdl)
  
  if ((!is.null(config$in.fold) && config$in.fold != -1) || config$holdout.validation) {
    preds = c(data.matrix(valid[, -which(names(valid) == 'target'), drop = F])  %*% mdl)
    score = config$eval.score.core(preds, valid$target)
    cat(date(), 'Validation score:', score, '\n')
    
    if ((!is.null(config$in.fold) && config$in.fold != -1)) {
      fnm = paste0(config$tmp.dir, '/cv-preds-', config$model.tag, '-', config$in.fold, '.RData')
      cat(date(), 'Saving valid preds to', fnm, '\n')
      save(preds, file = fnm)
    }
  }
  
  if ((!is.null(config$in.fold) && config$in.fold == -1)) {
    preds = c(data.matrix(test) %*% mdl)
    fnm = paste0(config$tmp.dir, '/test-preds-', config$model.tag, '.RData')
    cat(date(), 'Saving test preds to', fnm, '\n')
    save(preds, file = fnm)
  }
}

# Do stuff
# ==================================================================================================

if (config$mode == 'single') {
  cat(date(), 'Starting single mode\n')

  if (config$do.base.preprocess) {
    config$preprocess.raw1(config)
  }
  
  if (config$do.preprocess) {
    config$preprocess(config)
  }
  
  if (config$do.experiment) {
    config$experiment(config)
  }
  
  if (config$do.train) {
    ret = train(config)
  }
} else if (config$mode == 'cv') {
  cat(date(), 'Starting CV mode\n')

  if (config$do.base.preprocess) {
    config$preprocess.raw1(config)
  }
  
  if (config$do.preprocess) {
    config$preprocess(config)
  }
  
  if (config$do.train) {
    cross.validate(config)
  }
} else if (config$mode == 'cv.batch') {
  cat(date(), 'Batch CV mode\n')
  
  cv.batch(config)
}

if (config$do.submit) {
  generate.submission(config)
}

cat(date(), 'Done.\n')
