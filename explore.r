library(needs)
needs(data.table, caret, Hmisc, ComputeBackend)
source('../utils.r')

do.load    = T
do.cat1    = F
do.cat2    = F
do.qnum1   = F
do.ord2    = F
do.autoenc = T

if (do.load) {
  train = fread('input/train.csv', stringsAsFactors = T)
  
  # Reorder the levels of categoricals lexicographically (hexavegisimal)
  reorder.levels.hexv = function (x) {
    lvls = as.character(unique(x))
    lvls = lvls[order(nchar(lvls), tolower(lvls))]
    factor(x, levels = lvls)
  }
  cols = names(train)[sapply(train, is.factor)]
  for (j in cols) set(train, j = j, value = reorder.levels.hexv(train[[j]]))
}

if (do.cat1 || do.cat2 || do.ord2) {
  # Make all features categorical
  qnr = 30
  cat.features = names(train)[which(sapply(train, is.factor))]
  num.features = setdiff(names(train)[which(!sapply(train, is.factor))], c('loss', 'id'))
  for (j in num.features) set(train, j = j, value = cut2(train[[j]], g = min(qnr, uniqueN(train[[j]]))))
  y = train$loss
  train[, loss := NULL]
  train[, id   := NULL]
  cat.features = copy(names(train))
}

if (do.cat1) {
  # Marginal assoc testing
  cat1 = data.table(x = cat.features, p = unlist(mclapply(train[, cat.features, with = F], function(x, y) kruskal.test(y, x)$p.value, mc.cores = 8, y = y)))
  cat1[, p.adjusted := p.adjust(p, 'BH')]
  cat1 = cat1[order(p.adjusted)]
  save(cat1, file = 'assoc-cat1.RData')
}

# Pairwise association testing of all categoricals
if (do.cat2) {
  config = list()
  config$compute.backend = 'multicore'
  config$nr.cores = 8
  config$train = train
  config$train[, target := log(y + 200)]
  config$cat.pairs = data.frame(t(combn(cat.features, 2)), stringsAsFactors = F)
  names(config$cat.pairs) = c('x1', 'x2')
  config$cat.pairs = config$cat.pairs[sample(1:nrow(config$cat.pairs)), ]

  # Prune high cardinality categoricals
  nasty.features = setdiff(names(config$train)[which(sapply(config$train, uniqueN) > 50)], 'target')
  for (f in nasty.features) {
    config$train[[f]] = as.character(config$train[[f]])
    tbl = head(sort(table(config$train[[f]]), decreasing = T), 50)
    levels.to.keep = names(tbl[tbl > 10])
    config$train[!(config$train[[f]] %in% levels.to.keep), f] = NA
    config$train[[f]] = factor(config$train[[f]])
  }
  
  # We don't need that much data for this analysis, and it's pretty slow, so subsample
  config$train = config$train[1:25e3]
  
  test.job = function(config, core) {
    pair.idxs = compute.backend.balance(nrow(config$cat.pairs), config$nr.cores, core)
    nr.pairs.core = length(pair.idxs)
    if (nr.pairs.core == 0) {
      return (NULL)
    }
    
    pv = rep(NA, nr.pairs.core)

    for (ic in 1:nr.pairs.core) {
      cat(date(), ic, nr.pairs.core, '\n')
      i = pair.idxs[ic]
      f1 = config$cat.pairs$x1[i]
      f2 = config$cat.pairs$x2[i]
      pv[ic] = anova(lm(as.formula(paste('target ~', f1, '*', f2)), config$train))$`Pr(>F)`[3]
    }
    
    return (pv)
  }
  
  res = compute.backend.run(config, test.job, combine = c, package.dependencies = 'ComputeBackend')
  res[is.na(res)] = 1
  
  cat2 = cbind(as.data.table(config$cat.pairs), res)
  setnames(cat2, c('x1', 'x2', 'p12'))

  # NOTE: in this case the test I ran is a test of interaction above and beyond a marginal effect in
  # an unconstrained model. So no need to merge with the marginals. It's also easier to adjust for
  # multiplicity.
  cat2[, p12.adjusted := p.adjust(p12, 'BH')] # adjusting only for all pairwise tests, it's probably good enough
  cat2 = cat2[order(p12.adjusted)]
  
  # => hmm, problem is, about 1000 of them could easily be added... that's non trivial, because
  # many of these have a substantial amount of categories. It'll probably be best if I add them
  # as one stacked feature each rather than as these nasty creatures.
  
  save(cat2, file = 'assoc-cat2.RData')
}

if (do.qnum1) {
  # NOTE: this modifies train, reload it later if needed fresh
  qnum1 = data.table(x = num.features, p = unlist(mclapply(train[, num.features, with = F], function(x, y) chisq.test(x, y)$p.value, mc.cores = 8, y = y)))
  save(qnum1, file = 'assoc-qnum1.RData')
}

# Pairwise association testing of all categoricals, when treated as (hexavegisimal coded) ordinal
if (do.ord2) {
  config = list()
  config$compute.backend = 'multicore'
  config$nr.cores = 8
  config$cat.pairs = data.frame(t(combn(cat.features, 2)), stringsAsFactors = F)
  names(config$cat.pairs) = c('x1', 'x2')
  config$cat.pairs = config$cat.pairs[sample(1:nrow(config$cat.pairs)), ]

  train[, target := log(y + 200)]
  config$train = train[1:50e3] # We don't need that much data for this analysis, and it's pretty slow, so subsample
  config$train.idx = sample(nrow(config$train), floor(nrow(config$train)/2))

  test.job = function(config, core) {
    pair.idxs = compute.backend.balance(nrow(config$cat.pairs), config$nr.cores, core)
    nr.pairs.core = length(pair.idxs)
    if (nr.pairs.core == 0) {
      return (NULL)
    }
    
    dv = rep(NA, nr.pairs.core)

    for (ic in 1:nr.pairs.core) {
      cat(date(), ic, nr.pairs.core, '\n')
      
      i = pair.idxs[ic]
      f1 = config$cat.pairs$x1[i]
      f2 = config$cat.pairs$x2[i]
      x1 = as.numeric(config$train[[f1]])
      x2 = as.numeric(config$train[[f2]])
      x12 = as.numeric(interaction(config$train[[f1]], config$train[[f2]], drop = T, lex.order = config$lex.order))

      # Hmm.. a linear model GLRT or similar won't work here since the interaction is perfectly
      # linear in the components. So maybe I want to build a main effects model and a full model of
      # some sort, and measure the added predictive value (say via CV error) of the latter.
      
      dat = cbind(x1, x2, x12)
      xtrain = xgb.DMatrix(dat[ config$train.idx, 1, drop = F], label = config$train$target[ config$train.idx])
      xvalid = xgb.DMatrix(dat[-config$train.idx, 1, drop = F], label = config$train$target[-config$train.idx])
      mdl = xgb.train(data = xtrain, max_depth = 1, nrounds = 20, verbose = 0, nthread = 1)
      rmse.null1 = mean((predict(mdl, xvalid) - getinfo(xvalid, 'label'))^2)
      xtrain = xgb.DMatrix(dat[ config$train.idx, 2, drop = F], label = config$train$target[ config$train.idx])
      xvalid = xgb.DMatrix(dat[-config$train.idx, 2, drop = F], label = config$train$target[-config$train.idx])
      mdl = xgb.train(data = xtrain, max_depth = 1, nrounds = 20, verbose = 0, nthread = 1)
      rmse.null2 = mean((predict(mdl, xvalid) - getinfo(xvalid, 'label'))^2)
      xtrain = xgb.DMatrix(dat[ config$train.idx, 3, drop = F], label = config$train$target[ config$train.idx])
      xvalid = xgb.DMatrix(dat[-config$train.idx, 3, drop = F], label = config$train$target[-config$train.idx])
      mdl = xgb.train(data = xtrain, max_depth = 1, nrounds = 20, verbose = 0, nthread = 1)
      rmse.alte = mean((predict(mdl, xvalid) - getinfo(xvalid, 'label'))^2)
      
      dv[ic] = min(rmse.null1, rmse.null2) - rmse.alte
    }
    
    return (dv)
  }
  
  config$lex.order = F
  res = compute.backend.run(config, test.job, combine = c, package.dependencies = c('ComputeBackend', 'xgboost'))
  ord2 = cbind(as.data.table(config$cat.pairs), config$lex.order, res)
  
  config$lex.order = T
  res = compute.backend.run(config, test.job, combine = c, package.dependencies = c('ComputeBackend', 'xgboost'))
  ord2 = rbind(ord2, cbind(as.data.table(config$cat.pairs), config$lex.order, res))
  
  setnames(ord2, c('x1', 'x2', 'lex.order', 'd12'))
  ord2 = ord2[order(d12, decreasing = T)]
  
  print(ord2)
  
  save(ord2, file = 'assoc-ord2.RData')
}

if (do.autoenc) {
  needs(h2o, ggplot2)
  
  y = log(train$loss + 200)
  train[, loss := NULL]
  train[, id   := NULL]
  
  if (0) {
    cat.features = names(train)[which(sapply(train, is.factor))]
    for (j in cat.features) set(train, j = j, value = as.integer(train[[j]]) - 1)
    train = scale(train)
  } else if (0) {
    num.features = names(train)[-which(sapply(train, is.factor))]
    train = train[, num.features, with = F]
    train = scale(train)
  } else {
    cat.features = names(train)[which(sapply(train, is.factor))]
    train = train[, cat.features, with = F]
  }
    
  # FIXME how big can this be?
  train = train[1:1e4,]
  y = y[1:1e4]

  h2o.init(nthreads = 7, max_mem_size = '16G')
  train.h2o = as.h2o(train)
  
  # FIXME tune...
  m.aec = h2o.deeplearning(
    x = colnames(train),
    training_frame = train.h2o,
    autoencoder = T,
    activation = 'Rectifier',
    hidden = c(64, 32, 16, 2, 16, 32, 64),
    epochs = 500
  )
  
  deep.fea = as.data.frame(h2o.deepfeatures(m.aec, train.h2o, layer = 4))
  save(deep.fea, file = 'autoencoder-output.RData')
  
  # Plot what we get, for the trainset, against the true targets. Does it look informative?
  deep.fea.tr = data.frame(deep.fea, y)
  names(deep.fea.tr) = c('DF1', 'DF2', 'logloss')
  qplot(DF1, DF2, colour = logloss, data = deep.fea.tr) + scale_colour_gradient(limits = c(5, 11), low = 'black', high = 'white', space = 'Lab')
}

cat(date(), 'Done.\n')