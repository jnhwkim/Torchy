---------------------------------------------------------------------
-- Preprocessor for Movie Reivew Data v1.0 sentence polarity dataset comes
-- from the URL
-- http://www.cs.cornell.edu/people/pabo/movie-review-data .
--
-- Author: Jin-Hwa Kim (jnhwkim@snu.ac.kr)
-- License: BSD 3-Clause
---------------------------------------------------------------------

-- Options
if not opt then opt = {} end
opt.data_path = 'rt-polaritydata/rt-polarity'
opt.types = {'pos','neg'}
torch.manualSeed(123)

-- Dataset
local rtp = {}
rtp.word_to_idx = {}
rtp.idx_to_word = {}
rtp.word_freq = {}
rtp.nVocabulary = 0
rtp.nSample = 0
rtp.max_length = -1

print('Build vocabulary table..')
for _idx,t in pairs(opt.types) do
	local f = io.open(opt.data_path..'.'..t)
	while true do
		local line = f:read()
		if line == nil then break end
		rtp.nSample = rtp.nSample and rtp.nSample + 1 or 1
		local tokens = line:split(' ')
		rtp.max_length = math.max(rtp.max_length, #tokens)
		for _idx,w in pairs(tokens) do
			local widx = rtp.word_to_idx[w]
			local wfreq = rtp.word_freq[w]
			if not widx then
				rtp.nVocabulary = rtp.nVocabulary + 1
				rtp.word_to_idx[w] = rtp.nVocabulary
				widx = rtp.nVocabulary
				rtp.idx_to_word[widx] = w
			end
			rtp.word_freq[w] = wfreq and wfreq + 1 or 1
		end
	end
	f:close()
end

print('\nStatistics:')
print('number of samples = ', rtp.nSample)
print('number of vocabulary = ', rtp.nVocabulary)
print('maximum sentence length = ', rtp.max_length)

print('\nBuild data tensors..')
rtp.X = torch.IntTensor(rtp.nSample, rtp.max_length):zero()
rtp.y = torch.LongTensor(rtp.nSample):fill(1)  -- negative
rtp.y:narrow(1,1,rtp.nSample/2):fill(2)  -- positive

local i = 0
for _idx,t in pairs(opt.types) do
	local f = io.open(opt.data_path..'.'..t)
	while true do
		local line = f:read()
		if line == nil then break end
		i = i + 1
		local tokens = line:split(' ')
		for j,w in pairs(tokens) do
			if j > rtp.max_length then break end
			local widx = rtp.word_to_idx[w]
			local nTokens = math.min(rtp.max_length, #tokens)
			rtp.X[i][rtp.max_length+j-nTokens] = widx  -- right-aligned population
		end
	end
	f:close()
end

collectgarbage()

print('\nTensor Information:')
require 'torchx'  -- provides `find`
print('zero ratio = ', #torch.find(rtp.X, 0)/rtp.X:nElement())

print('\nMovie Review Dataset loaded.\n')

-- Return results
return rtp