--[[ params:
   basepath: a folder containing the three txt files
--]]

require 'datasources.datasource'
require 'paths'

local PTBCharDatasource, parent = torch.class('PTBCharDatasource', 'TextDatasource')

function PTBCharDatasource:__init(params)
   parent.__init(self)
   params = params or {}
   params.basepath = params.basepath or '/misc/vlgscratch2/LecunGroup/michael/datasets/pentreebank'
   
   -- load txt
   local used_chars = {}
   local function loadFile(filename)
      local dataset = {}
      for line in io.lines(filename) do
	 local goodline = line:gsub(" ", "")
	 if #goodline > 0 then
	    local tens = torch.ByteTensor(#goodline)
	    for i = 1,#goodline do
	       local b = string.byte(goodline, i)
	       tens[i] = b
	       used_chars[b] = true
	    end
	    dataset[1+#dataset] = tens
	 end
      end
      return dataset
   end
   self.data = {
      train = loadFile(paths.concat(params.basepath, 'ptb.char.train.txt')),
      test = loadFile(paths.concat(params.basepath, 'ptb.char.test.txt')),
      valid = loadFile(paths.concat(params.basepath, 'ptb.char.valid.txt'))}
   self.tokens = {}
   for k,v in pairs(used_chars) do
      self.tokens[1+#self.tokens] = k
   end
   self.tokens[1+#self.tokens] = 0 -- End of sentence
   self.nTokens = #self.tokens
   self.revTokens = {}
   for i = 1, #self.tokens do
      self.revTokens[self.tokens[i] ] = i
   end

   -- process them
   for set, data in pairs(self.data) do
      for i = 1, #data do
	 for j = 1, data[i]:size(1) do
	    data[i][j] = self.revTokens[data[i][j] ]
	 end
      end
   end
end

function PTBCharDatasource:nextBatch(batchSize, set)
   assert((batchSize ~= nil) and (self.data[set] ~= nil))
   local indices, maxlen = {}, 0
   for i = 1, batchSize do
      indices[i] = torch.random(#self.data[set])
      maxlen = math.max(maxlen, self.data[set][indices[i] ]:size(1))
   end
   self.output_cpu:resize(maxlen, batchSize):fill(self.revTokens[0])
   for i = 1, batchSize do
      local sample = self.data[set][indices[i] ]
      self.output_cpu[{{1,sample:size(1)},i}]:copy(sample)
   end
   return self:typeResults(self.output_cpu)
end

function PTBCharDatasource:orderedIterator(batchSize, set)
   assert((batchSize ~= nil) and (self.data[set] ~= nil))
   local idx = 1
   return function()
      local indices, maxlen = {}, 0
      for i = 1, batchSize do
	 if idx <= #self.data[set] then
	    indices[i] = idx
	    maxlen = math.max(maxlen, self.data[set][idx]:size(1))
	 else
	    break
	 end
	 idx = idx + 1
      end
      if #indices == 0 then
	 return nil
      else
	 self.output_cpu:resize(maxlen, #indices):fill(self.revTokens[0])
	 for i = 1, #indices do
	    local sample = self.data[set][indices[i] ]
	    self.output_cpu[{{1,sample:size(1)}, i}]:copy(sample)
	 end
	 return self:typeResults(self.output_cpu)
      end
   end
end