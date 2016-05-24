---------------------------------------------------------------------
-- Training code for Movie Reivew Data v1.0 sentence polarity dataset comes
-- from the URL
-- http://www.cs.cornell.edu/people/pabo/movie-review-data .
--
-- Author: Jin-Hwa Kim (jnhwkim@snu.ac.kr)
-- License: BSD 3-Clause
---------------------------------------------------------------------

require 'rnn'
require 'optim'

-- Training Options
opt = {}
opt.cv = 10
opt.hiddenSize = 80
opt.batchSize = 64
opt.learningRate = .001
opt.maxEpoch = 5
opt.cuda = false
torch.manualSeed(123)

if opt.cuda then 
   require 'cutorch'
   require 'cunn'
end

MR = require 'MR'  -- load Movie Review dataset
if opt.cuda then
   MR.X = MR.X:cuda()
   MR.y = MR.y:cuda()
end

-- Cross Validation
function cv(nSample, n)
   local idx = torch.randperm(nSample)
   local ssize = math.floor(nSample/n)
   local idx_train = torch.LongTensor(n, nSample-ssize)
   local idx_val = torch.LongTensor(n, ssize)
   for i=1,n do
      local pos1, pos2 = ssize*(i-1)+1, ssize*i+1
      if i ~= 1 then 
         idx_train[i][{{1,pos1-1}}]:copy(idx[{{1,pos1-1}}])
      end; if i ~= n then 
         idx_train[i][{{pos1,idx_train:size(2)}}]:copy(idx[{{pos2,idx:size(1)}}])
      end
      idx_val[i]:copy(idx:narrow(1,pos1,ssize))
   end
   return idx_train, idx_val
end
idx_train, idx_val = cv(MR.nSample, opt.cv)  -- 10-fold cross validation

-- Model definition
rnnModule = nn.GRU(opt.hiddenSize, opt.hiddenSize, nil, .25, true):maskZero(1)
model = nn.Sequential()
   :add(nn.LookupTableMaskZero(MR.nVocabulary, opt.hiddenSize))
   :add(nn.SplitTable(2))
   :add(nn.Sequencer(rnnModule))
   :add(nn.SelectTable(-1))
   :add(nn.Dropout(.5))
   :add(nn.Linear(opt.hiddenSize, 2))

-- Criterion
criterion = nn.CrossEntropyCriterion()

if opt.cuda then
   model = model:cuda()
   criterion = criterion:cuda()
end
w,dw = model:getParameters()

function JDJ(_w)
   w:copy(_w)
   dw:zero()
   local output = model:forward(X)
   J = criterion:forward(output, y)
   local dy = criterion:backward(output, y)
   model:backward(X, dy)
   dw:clamp(-10,10)  -- Pascanu et al., 2013
   return J,dw
end

function evaluate(k)
   local X = MR.X:index(1,idx_val[k])
   local y = MR.y:index(1,idx_val[k])
   model:evaluate()  -- for dropout
   output = model:forward(X)
   _max,pred = torch.max(output,2)
   local diff = pred-y:resizeAs(pred)
   require 'torchx'
   model:training()
   return #torch.find(diff, 0) / y:nElement()
end


nSample = idx_train:size(2)

for k=1,opt.cv do
   local config={}
   config.learningRate=opt.learningRate
   config.maxIter=opt.maxEpoch
   config.update_grad_per_n_batches=1
   w:uniform(-.08,.08)  -- initialize parameters
   config.winit = w
   local accuracy = evaluate(k)
   print(string.format('CV-%d Accuracy = %.2f', k, accuracy))
   for epoch=1,opt.maxEpoch do
      for i=1,nSample,opt.batchSize do
         xlua.progress(i, nSample)
         local j=math.min(i+opt.batchSize-1,nSample)
         X = MR.X:index(1,idx_train[k][{{i,j}}])
         y = MR.y:index(1,idx_train[k][{{i,j}}])
         optim.adam(JDJ, config.winit, config)
      end
      xlua.progress(nSample, nSample)  -- done
      print(string.format('Epoch#%2d\tlearningRate = ', epoch), config.learningRate, 
            ', J = ', J)
      local accuracy = evaluate(k)
      print(string.format('CV-%d Accuracy = %.4f', k, accuracy))
      collectgarbage()
      if 0 == epoch % 2 then config.learningRate = config.learningRate / 2 end
   end
end